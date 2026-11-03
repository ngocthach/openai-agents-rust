//! Output-schema, timeout-output, converter and configuration cases of Python's
//! `tests/test_programmatic_tool_calling.py` (source lines 348-890 and the late-timeout pair at
//! 544).
//!
//! Python's `function_tool` validates `allowed_callers` and output options when the tool is
//! built. Rust has no such constructor step, so the same checks run when the tool is put in a
//! request: through `OpenAIResponsesModel::build_request_body` for the converter cases, and
//! through `Runner::run` (before any model call) for the construction cases.

use std::sync::Arc;

use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::model_settings::{ModelSettings, ToolChoice};
use openai_agents_sdk::models::OpenAIClient;
use openai_agents_sdk::models::interface::{ModelRequest, ToolSpec};
use openai_agents_sdk::models::openai_chatcompletions::OpenAIChatCompletionsModel;
use openai_agents_sdk::models::openai_responses::OpenAIResponsesModel;
use openai_agents_sdk::prompts::Prompt;
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::tool::{
    ApplyPatchTool, CodeInterpreterTool, CustomTool, FunctionTool, HostedMCPTool,
    ProgrammaticToolCallingTool, ShellTool, Tool, ToolSearchTool,
};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{Agent, AgentsError, RunContext, Runner};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

const PROGRAM_CALL_ID: &str = "call_program";
const FUNCTION_CALL_ID: &str = "call_lookup";

fn program_caller() -> Value {
    json!({"type": "program", "caller_id": PROGRAM_CALL_ID})
}

#[derive(Debug, Deserialize, JsonSchema, Serialize, PartialEq)]
struct InventoryOutput {
    sku: String,
    available_units: i64,
}

#[derive(Deserialize, JsonSchema)]
struct SkuArgs {
    #[allow(dead_code)]
    sku: String,
}

fn lookup_with_output(callers: Option<Vec<String>>) -> FunctionTool<()> {
    let tool = FunctionTool::new(
        "lookup_inventory",
        "",
        |_ctx: ToolContext<()>, args: SkuArgs| async move {
            Ok::<_, AgentsError>(json!({"sku": args.sku, "available_units": 42}))
        },
    );
    match callers {
        Some(callers) => tool.with_allowed_callers(callers),
        None => tool,
    }
}

fn invoke_context(tool_name: &str) -> ToolContext<()> {
    ToolContext::direct(RunContext::new(()), "agent", tool_name, "{}")
}

fn spec(tool: &Tool<()>) -> ToolSpec {
    match tool {
        Tool::Function(f) => ToolSpec::Function(f.to_spec()),
        other => {
            let (param, includes) = other.to_responses_param();
            ToolSpec::Hosted { param, includes }
        }
    }
}

fn responses_body(tools: &[Tool<()>], choice: Option<&str>) -> openai_agents_sdk::Result<Value> {
    let model = OpenAIResponsesModel::new("gpt-test", OpenAIClient::new("sk-test"));
    let request = ModelRequest {
        tools: tools.iter().map(spec).collect(),
        model_settings: ModelSettings {
            tool_choice: choice.map(ToolChoice::from),
            ..Default::default()
        },
        ..Default::default()
    };
    model.build_request_body(&request, false)
}

fn user_error(result: openai_agents_sdk::Result<impl std::fmt::Debug>) -> String {
    match result {
        Err(AgentsError::UserError { message, .. }) => message,
        Err(other) => panic!("expected UserError, got {other}"),
        Ok(value) => panic!("expected UserError, got {value:?}"),
    }
}

fn function_output_items(result: &openai_agents_sdk::RunResult<()>) -> Vec<Value> {
    result
        .new_items
        .iter()
        .filter_map(|item| match &item.kind {
            RunItemKind::ToolCallOutput { raw_item, .. }
                if raw_item["type"] == "function_call_output" =>
            {
                Some(raw_item.clone())
            }
            _ => None,
        })
        .collect()
}

// ---------------------------------------------------------------------------------------------
// Output type options
// ---------------------------------------------------------------------------------------------

#[test]
fn function_tool_output_type_override_and_raw_schema_are_mutually_exclusive() {
    // Python: output_type alone infers the schema; combined with output_json_schema it raises.
    let tool = lookup_with_output(Some(vec!["programmatic".into()]))
        .with_output_type::<InventoryOutput>()
        .unwrap();
    assert_eq!(
        tool.output_json_schema.as_ref().unwrap()["title"],
        json!("InventoryOutput")
    );

    let message = user_error(
        lookup_with_output(None)
            .with_output_type::<InventoryOutput>()
            .unwrap()
            .with_output_json_schema(json!({"type": "object"})),
    );
    assert!(message.contains("cannot both be provided"), "{message}");

    let message = user_error(
        lookup_with_output(None)
            .with_output_json_schema(json!({
                "type": "object",
                "properties": {"sku": {"type": "string"}},
                "required": ["sku"],
                "additionalProperties": false,
            }))
            .unwrap()
            .with_output_type::<InventoryOutput>(),
    );
    assert!(message.contains("cannot both be provided"), "{message}");

    let message = user_error(lookup_with_output(None).with_output_type::<String>());
    assert!(
        message.contains("output_type must define a strict JSON object"),
        "{message}"
    );
}

#[tokio::test]
async fn schema_backed_function_tool_accepts_conforming_custom_error_output() {
    // A failure handler may return a value that matches the declared output schema.
    let tool = FunctionTool::new(
        "failing_tool",
        "",
        |_ctx: ToolContext<()>, _args: Value| async move {
            Err::<Value, _>(AgentsError::user("inventory unavailable"))
        },
    )
    .with_output_type::<InventoryOutput>()
    .unwrap()
    .with_failure_error_function(|_ctx, _err| {
        json!({"sku": "ERROR", "available_units": 0}).to_string()
    });

    let output = tool
        .invoke(invoke_context("failing_tool"), "{}".into())
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&output.to_model_string()).unwrap(),
        json!({"sku": "ERROR", "available_units": 0})
    );
}

#[tokio::test]
async fn schema_backed_programmatic_tool_accepts_conforming_custom_timeout_output() {
    let m = Arc::new(ScriptedModel::new());
    m.extend(vec![
        ScriptedStep::output(vec![
            OutputItem::from_value(json!({
                "type": "program", "id": "program_item", "call_id": PROGRAM_CALL_ID,
                "code": "lookup_inventory(sku=\"A-1\")", "fingerprint": "fingerprint",
            })),
            OutputItem::from_value(json!({
                "type": "function_call", "id": "function_item", "call_id": FUNCTION_CALL_ID,
                "name": "lookup_inventory", "arguments": "{\"sku\":\"A-1\"}",
                "caller": program_caller(),
            })),
        ]),
        ScriptedStep::output(vec![
            OutputItem::from_value(json!({
                "type": "program_output", "id": "program_output_item",
                "call_id": PROGRAM_CALL_ID, "result": "{}", "status": "completed",
            })),
            assistant_message("timeout handled"),
        ]),
    ])
    .unwrap();

    let tool = FunctionTool::new(
        "lookup_inventory",
        "",
        |_ctx: ToolContext<()>, args: SkuArgs| async move {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            Ok::<_, AgentsError>(json!({"sku": args.sku, "available_units": 42}))
        },
    )
    .with_output_type::<InventoryOutput>()
    .unwrap()
    .with_allowed_callers(vec!["programmatic".into()])
    .with_timeout(std::time::Duration::from_millis(10))
    .with_timeout_error_function(|_ctx, _err| {
        json!({"sku": "TIMEOUT", "available_units": 0}).to_string()
    });

    let agent = Agent::new("inventory")
        .with_model_instance(m)
        .with_tool(ProgrammaticToolCallingTool)
        .with_tool(tool);
    let result = Runner::run(agent, "Check inventory").await.unwrap();

    assert_eq!(
        result.final_output_text().as_deref(),
        Some("timeout handled")
    );
    let outputs = function_output_items(&result);
    assert_eq!(outputs.len(), 1);
    assert_eq!(
        serde_json::from_str::<Value>(outputs[0]["output"].as_str().unwrap()).unwrap(),
        json!({"sku": "TIMEOUT", "available_units": 0})
    );
}

#[tokio::test]
async fn schema_backed_function_output_rejects_plain_error_text() {
    // Python's first case: a plain error string from the failure handler does not match the
    // declared output schema. The second case (an unvalidated `{"type": "object"}` schema passed
    // straight to the item helper) has no Rust equivalent; see the report.
    let tool = FunctionTool::new(
        "lookup_inventory",
        "",
        |_ctx: ToolContext<()>, _args: Value| async move {
            Err::<Value, _>(AgentsError::user("inventory unavailable"))
        },
    )
    .with_output_type::<InventoryOutput>()
    .unwrap()
    .with_failure_error_function(|_ctx, _err| "inventory unavailable".to_string());

    let err = tool
        .invoke(invoke_context("lookup_inventory"), "{}".into())
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("does not match its declared output"),
        "{err}"
    );
}

#[tokio::test]
async fn function_tool_serializes_typed_output_with_schema_aliases() {
    // Python's `validation_alias="inputUnits"` / `serialization_alias="availableUnits"`.
    #[derive(Debug, Deserialize, JsonSchema, Serialize)]
    struct AliasedInventoryOutput {
        sku: String,
        #[serde(rename(serialize = "availableUnits", deserialize = "inputUnits"))]
        #[serde(alias = "availableUnits")]
        available_units: i64,
    }

    let tool = FunctionTool::new(
        "aliased_tool",
        "",
        |_ctx: ToolContext<()>, _args: Value| async move {
            Ok::<_, AgentsError>(json!({"sku": "A-1", "availableUnits": 42}))
        },
    )
    .with_output_type::<AliasedInventoryOutput>()
    .unwrap()
    .with_allowed_callers(vec!["programmatic".into()]);

    let schema = tool.output_json_schema.clone().expect("schema");
    assert!(
        schema["properties"].get("availableUnits").is_some(),
        "{schema}"
    );

    let output = tool
        .invoke(invoke_context("aliased_tool"), "{}".into())
        .await
        .unwrap();
    assert_eq!(
        serde_json::from_str::<Value>(&output.to_model_string()).unwrap(),
        json!({"sku": "A-1", "availableUnits": 42})
    );
}

// ---------------------------------------------------------------------------------------------
// Converter: allowed_callers and programmatic configuration
// ---------------------------------------------------------------------------------------------

#[test]
fn responses_converter_serializes_allowed_callers_for_other_eligible_tools() {
    let shell = ShellTool::new(|_req| async { Ok::<_, AgentsError>("ok".to_string()) })
        .with_allowed_callers(vec!["programmatic".into()]);
    let apply = ApplyPatchTool::new(FakeEditor)
        .with_allowed_callers(vec!["direct".into(), "programmatic".into()]);
    let custom = CustomTool::new(
        "custom",
        "Custom tool",
        |_ctx: ToolContext<()>, _input: String| async { Ok::<_, AgentsError>("ok".to_string()) },
    )
    .with_allowed_callers(vec!["programmatic".into()]);

    let body = responses_body(
        &[
            Tool::from(ProgrammaticToolCallingTool),
            Tool::from(shell),
            Tool::from(apply),
            Tool::from(custom),
        ],
        None,
    )
    .unwrap();
    let tools = body["tools"].as_array().unwrap();
    assert_eq!(tools[1]["allowed_callers"], json!(["programmatic"]));
    assert_eq!(
        tools[2]["allowed_callers"],
        json!(["direct", "programmatic"])
    );
    assert_eq!(tools[3]["allowed_callers"], json!(["programmatic"]));
}

#[tokio::test]
async fn tool_construction_rejects_invalid_allowed_callers() {
    // Python checks the list when each tool is built. Rust checks it when the tool is put in a
    // request, so each case runs the agent against a model that must not be called.
    let invalid: [Vec<&str>; 3] = [vec![], vec!["direct", "direct"], vec!["unsupported"]];
    for callers in invalid {
        let owned: Vec<String> = callers.iter().map(|c| c.to_string()).collect();
        let shell = ShellTool::new(|_req| async { Ok::<_, AgentsError>("ok".to_string()) })
            .with_allowed_callers(owned.clone());
        let hosted = HostedMCPTool::new(json!({
            "type": "mcp", "server_label": "inventory", "server_url": "https://example.com/mcp",
            "allowed_callers": owned.clone(),
        }));
        let code = CodeInterpreterTool {
            tool_config: json!({
                "type": "code_interpreter", "container": "auto", "allowed_callers": owned.clone(),
            }),
        };
        let function = lookup_with_output(Some(owned));

        for tool in [
            Tool::from(function),
            Tool::from(shell),
            Tool::from(hosted),
            Tool::from(code),
        ] {
            let model = Arc::new(ScriptedModel::new());
            let agent = Agent::new("inventory")
                .with_model_instance(model.clone())
                .with_tool(tool);
            let err = Runner::run(agent, "Check").await.unwrap_err();
            match err {
                AgentsError::UserError { message, .. } => {
                    assert!(
                        message.contains("allowed_callers"),
                        "{callers:?}: {message}"
                    );
                }
                other => panic!("{callers:?}: expected UserError, got {other}"),
            }
            assert!(model.calls().is_empty(), "{callers:?}: model was called");
        }
    }
}

#[test]
fn responses_converter_rejects_incomplete_programmatic_configuration() {
    let programmatic_only = lookup_with_output(Some(vec!["programmatic".into()]));
    let message = user_error(responses_body(&[Tool::from(programmatic_only)], None));
    assert!(
        message.contains("requires ProgrammaticToolCallingTool"),
        "{message}"
    );

    let message = user_error(responses_body(&[], Some("programmatic_tool_calling")));
    assert!(
        message.contains("requires ProgrammaticToolCallingTool"),
        "{message}"
    );

    let message = user_error(responses_body(
        &[Tool::from(ProgrammaticToolCallingTool)],
        None,
    ));
    assert!(message.contains("requires at least one tool"), "{message}");

    let message = user_error(responses_body(
        &[
            Tool::from(ProgrammaticToolCallingTool),
            Tool::from(ProgrammaticToolCallingTool),
        ],
        None,
    ));
    assert!(
        message.contains("Only one ProgrammaticToolCallingTool"),
        "{message}"
    );
}

#[test]
fn responses_converter_accepts_mixed_or_tool_search_managed_configuration() {
    // A function open to both callers is valid without a programmatic tool in the request.
    let mixed = lookup_with_output(Some(vec!["direct".into(), "programmatic".into()]));
    let body = responses_body(&[Tool::from(mixed)], None).unwrap();
    assert_eq!(
        body["tools"][0]["allowed_callers"],
        json!(["direct", "programmatic"])
    );

    // A prompt-managed request may use tool search as the programmatic caller surface.
    let model = OpenAIResponsesModel::new("gpt-test", OpenAIClient::new("sk-test"));
    let request = ModelRequest {
        tools: vec![
            spec(&Tool::from(ProgrammaticToolCallingTool)),
            spec(&Tool::from(ToolSearchTool::default())),
        ],
        prompt: Some(Prompt::new("pmpt_123")),
        ..Default::default()
    };
    let body = model.build_request_body(&request, false).unwrap();
    assert_eq!(
        body["tools"],
        json!([{"type": "programmatic_tool_calling"}, {"type": "tool_search"}])
    );
}

#[test]
fn chat_completions_rejects_programmatic_tool_configuration() {
    let model = OpenAIChatCompletionsModel::new("gpt-test", OpenAIClient::new("sk-test"));
    let request_with = |tools: Vec<Tool<()>>, choice: Option<&str>| ModelRequest {
        tools: tools.iter().map(spec).collect(),
        model_settings: ModelSettings {
            tool_choice: choice.map(ToolChoice::from),
            ..Default::default()
        },
        ..Default::default()
    };

    let function = lookup_with_output(Some(vec!["programmatic".into()]));
    let message = user_error(
        model.build_request_body(&request_with(vec![Tool::from(function)], None), false),
    );
    assert!(
        message.contains("only supported with OpenAI Responses models"),
        "{message}"
    );

    let message = user_error(model.build_request_body(
        &request_with(vec![], Some("programmatic_tool_calling")),
        false,
    ));
    assert!(message.contains("programmatic_tool_calling"), "{message}");

    let message = user_error(model.build_request_body(
        &request_with(vec![Tool::from(ProgrammaticToolCallingTool)], None),
        false,
    ));
    assert!(
        message.contains("Hosted tools are not supported"),
        "{message}"
    );
}

// ---------------------------------------------------------------------------------------------
// Timeouts and failure output on the runner
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn runner_validates_only_user_output_after_late_tool_timeout() {
    // Python monkeypatches `asyncio.wait_for` so the deadline fires after the task settles. The
    // observable path is the tool's own RuntimeError reaching the formatter: the default text
    // for a direct tool, and a schema check on the custom failure text otherwise.
    let expected_error = "An error occurred while running the tool. Please try again.";
    for custom_failure in [false, true] {
        let mut tool = FunctionTool::new(
            "lookup_inventory",
            "",
            |_ctx: ToolContext<()>, _args: Value| async move {
                Err::<Value, _>(AgentsError::user("inventory unavailable"))
            },
        )
        .with_output_type::<InventoryOutput>()
        .unwrap()
        .with_timeout(std::time::Duration::from_secs(1));
        if custom_failure {
            tool = tool.with_failure_error_function(move |_ctx, _err| expected_error.to_string());
        }
        let m = Arc::new(ScriptedModel::new());
        m.extend(vec![
            ScriptedStep::output(vec![OutputItem::from_value(json!({
                "type": "function_call", "id": "function_item", "call_id": FUNCTION_CALL_ID,
                "name": "lookup_inventory", "arguments": "{}",
            }))]),
            ScriptedStep::output(vec![assistant_message("inventory lookup failed")]),
        ])
        .unwrap();
        let agent = Agent::new("inventory")
            .with_model_instance(m.clone())
            .with_tool(tool);
        let result = Runner::run(agent, "Check inventory").await;

        if custom_failure {
            let message = user_error(result);
            assert!(
                message.contains("does not match its declared output"),
                "{message}"
            );
            assert_eq!(m.calls().len(), 1);
            continue;
        }
        let result = result.unwrap();
        assert_eq!(
            result.final_output_text().as_deref(),
            Some("inventory lookup failed")
        );
        let outputs = function_output_items(&result);
        assert_eq!(outputs.len(), 1);
        assert_eq!(outputs[0]["output"], json!(expected_error));
    }
}

#[tokio::test]
async fn runner_preserves_direct_default_timeout_for_schema_backed_tool() {
    let m = Arc::new(ScriptedModel::new());
    m.extend(vec![
        ScriptedStep::output(vec![OutputItem::from_value(json!({
            "type": "function_call", "id": "function_item", "call_id": FUNCTION_CALL_ID,
            "name": "lookup_inventory", "arguments": "{\"sku\":\"A-1\"}",
        }))]),
        ScriptedStep::output(vec![assistant_message("timed out")]),
    ])
    .unwrap();
    let tool = FunctionTool::new(
        "lookup_inventory",
        "",
        |_ctx: ToolContext<()>, args: SkuArgs| async move {
            tokio::time::sleep(std::time::Duration::from_millis(200)).await;
            Ok::<_, AgentsError>(json!({"sku": args.sku, "available_units": 42}))
        },
    )
    .with_output_type::<InventoryOutput>()
    .unwrap()
    .with_allowed_callers(vec!["direct".into(), "programmatic".into()])
    .with_timeout(std::time::Duration::from_millis(10));
    let agent = Agent::new("inventory")
        .with_model_instance(m)
        .with_tool(tool);

    let result = Runner::run(agent, "Check inventory").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("timed out"));
    let outputs = function_output_items(&result);
    assert_eq!(outputs.len(), 1);
    let output = outputs[0]["output"].as_str().unwrap();
    assert!(output.to_lowercase().contains("timed out"), "{output}");
}

// ---------------------------------------------------------------------------------------------
// Function output and caller
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn function_output_preserves_caller_and_uses_declared_json_schema() {
    // A program-owned call keeps its caller on the output; the structured result is JSON text.
    let m = Arc::new(ScriptedModel::new());
    m.extend(vec![
        ScriptedStep::output(vec![
            OutputItem::from_value(json!({
                "type": "program", "id": "program_item", "call_id": PROGRAM_CALL_ID,
                "code": "lookup_inventory(sku=\"A-1\")", "fingerprint": "fingerprint",
            })),
            OutputItem::from_value(json!({
                "type": "function_call", "id": "function_item", "call_id": FUNCTION_CALL_ID,
                "name": "units_tool", "arguments": "{}", "caller": program_caller(),
            })),
        ]),
        ScriptedStep::output(vec![
            OutputItem::from_value(json!({
                "type": "program_output", "id": "program_output_item",
                "call_id": PROGRAM_CALL_ID, "result": "{}", "status": "completed",
            })),
            assistant_message("done"),
        ]),
    ])
    .unwrap();
    let tool = FunctionTool::new(
        "units_tool",
        "",
        |_ctx: ToolContext<()>, _args: Value| async move {
            Ok::<_, AgentsError>(json!({"sku": "A-1", "units": [1, 2]}))
        },
    )
    .with_allowed_callers(vec!["programmatic".into()]);
    let agent = Agent::new("inventory")
        .with_model_instance(m)
        .with_tool(ProgrammaticToolCallingTool)
        .with_tool(tool);

    let result = Runner::run(agent, "Check").await.unwrap();
    let outputs = function_output_items(&result);
    assert_eq!(outputs.len(), 1);
    assert_eq!(
        serde_json::from_str::<Value>(outputs[0]["output"].as_str().unwrap()).unwrap(),
        json!({"sku": "A-1", "units": [1, 2]})
    );
    assert_eq!(outputs[0]["caller"], program_caller());
}

/// Same file-free fixture as the shelltools tests: a no-op editor for apply-patch tools.
struct FakeEditor;

#[async_trait::async_trait]
impl openai_agents_sdk::editor::ApplyPatchEditor for FakeEditor {
    async fn create_file(
        &self,
        _operation: &openai_agents_sdk::editor::ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<openai_agents_sdk::editor::ApplyPatchResult> {
        Ok(Default::default())
    }
    async fn update_file(
        &self,
        _operation: &openai_agents_sdk::editor::ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<openai_agents_sdk::editor::ApplyPatchResult> {
        Ok(Default::default())
    }
    async fn delete_file(
        &self,
        _operation: &openai_agents_sdk::editor::ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<openai_agents_sdk::editor::ApplyPatchResult> {
        Ok(Default::default())
    }
}
