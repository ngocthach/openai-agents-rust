//! Program-item ownership cases of Python's `tests/test_programmatic_tool_calling.py`.
//!
//! Python drives `process_model_response` directly with `existing_items` and
//! `server_manages_conversation`. Rust keeps that function private, so each case is driven
//! through `Runner::run` with a scripted model: an `existing_items` parent becomes an earlier
//! turn of the same run, and server-managed conversation becomes `previous_response_id` plus a
//! submitted input delta. Python's `as_mapping` twins use model objects while Rust carries
//! every item as a JSON `Value`, so the twin of a case is the same test with the same input.

use std::sync::Arc;

use openai_agents_sdk::editor::{ApplyPatchOperation, ApplyPatchResult};
use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::testing::{ScriptedModel, ScriptedStep, assistant_message};
use openai_agents_sdk::tool::{
    ApplyPatchTool, CodeInterpreterTool, CustomTool, FunctionTool, HostedMCPTool,
    ProgrammaticToolCallingTool, ShellTool, ToolSearchTool,
};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{Agent, AgentsError, RunResult, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

const PROGRAM_CALL_ID: &str = "call_program";
const FUNCTION_CALL_ID: &str = "call_lookup";

fn program_caller() -> Value {
    json!({"type": "program", "caller_id": PROGRAM_CALL_ID})
}

fn program_item() -> OutputItem {
    OutputItem::from_value(json!({
        "type": "program", "id": "program_item", "call_id": PROGRAM_CALL_ID,
        "code": "lookup_inventory(sku=\"A-1\")", "fingerprint": "fingerprint",
    }))
}

fn lookup_call() -> OutputItem {
    OutputItem::from_value(json!({
        "type": "function_call", "id": "function_item", "call_id": FUNCTION_CALL_ID,
        "name": "lookup_inventory", "arguments": "{\"sku\":\"A-1\"}",
        "caller": program_caller(),
    }))
}

fn program_output(status: &str) -> OutputItem {
    OutputItem::from_value(json!({
        "type": "program_output", "id": "program_output_item", "call_id": PROGRAM_CALL_ID,
        "result": "{\"sku\":\"A-1\",\"available_units\":42}", "status": status,
    }))
}

fn text(message: &str) -> OutputItem {
    assistant_message(message)
}

fn model(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    let model = Arc::new(ScriptedModel::new());
    model
        .extend(steps.into_iter().map(ScriptedStep::output).collect())
        .unwrap();
    model
}

#[derive(Deserialize, JsonSchema)]
struct SkuArgs {
    #[allow(dead_code)]
    sku: String,
}

/// `lookup_inventory`; `callers` is the `allowed_callers` list, `None` for the default.
fn lookup_tool(callers: Option<&[&str]>) -> FunctionTool<()> {
    let tool = FunctionTool::new(
        "lookup_inventory",
        "",
        |_ctx: ToolContext<()>, args: SkuArgs| async move {
            Ok::<_, AgentsError>(json!({"sku": args.sku, "available_units": 42}))
        },
    );
    match callers {
        Some(callers) => tool.with_allowed_callers(callers.iter().map(|c| c.to_string()).collect()),
        None => tool,
    }
}

fn inventory_agent(model: Arc<ScriptedModel>) -> Agent<()> {
    Agent::new("inventory").with_model_instance(model)
}

fn with_programmatic(model: Arc<ScriptedModel>, callers: Option<&[&str]>) -> Agent<()> {
    inventory_agent(model)
        .with_tool(ProgrammaticToolCallingTool)
        .with_tool(lookup_tool(callers))
}

fn behavior_error(result: openai_agents_sdk::Result<RunResult<()>>) -> String {
    match result {
        Err(AgentsError::ModelBehaviorError { message, .. }) => message,
        Err(other) => panic!("expected ModelBehaviorError, got {other}"),
        Ok(_) => panic!("expected ModelBehaviorError, run succeeded"),
    }
}

fn raw_types(result: &RunResult<()>) -> Vec<String> {
    result
        .new_items
        .iter()
        .filter_map(|item| match &item.kind {
            RunItemKind::ToolCall { raw_item } | RunItemKind::ToolCallOutput { raw_item, .. } => {
                raw_item
                    .get("type")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            }
            _ => None,
        })
        .collect()
}

fn count_outputs(result: &RunResult<()>) -> usize {
    result
        .new_items
        .iter()
        .filter(|item| matches!(item.kind, RunItemKind::ToolCallOutput { .. }))
        .count()
}

fn ordered_subsequence(haystack: &[String], needle: &[&str]) -> bool {
    let mut idx = 0;
    for h in haystack {
        if idx < needle.len() && h == needle[idx] {
            idx += 1;
        }
    }
    idx == needle.len()
}

// ---------------------------------------------------------------------------------------------
// Ordering and parent/child structure
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn process_model_response_keeps_program_items_in_order() {
    // Python processes one response: program, function call, incomplete output. The Rust run
    // also executes the call, so its output follows the call; the program items keep their order.
    let m = model(vec![
        vec![program_item(), lookup_call(), program_output("incomplete")],
        vec![program_output("completed"), text("done")],
    ]);
    let result = Runner::run(with_programmatic(m, Some(&["programmatic"])), "Check")
        .await
        .unwrap();
    let types = raw_types(&result);
    assert!(
        ordered_subsequence(&types, &["program", "function_call", "program_output"]),
        "{types:?}"
    );
    assert!(
        types.first().map(String::as_str) == Some("program"),
        "{types:?}"
    );
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
}

#[tokio::test]
async fn process_model_response_rejects_program_without_valid_call_id() {
    // Python parametrizes a missing `call_id` and an empty one (two cases).
    for call_id in [None, Some("")] {
        let mut program = json!({
            "type": "program", "id": "program_item",
            "code": "return 42", "fingerprint": "fingerprint",
        });
        if let Some(call_id) = call_id {
            program["call_id"] = json!(call_id);
        }
        let m = model(vec![vec![OutputItem::from_value(program)]]);
        let agent = Agent::new("inventory")
            .with_model_instance(m)
            .with_tool(ProgrammaticToolCallingTool);
        let message = behavior_error(Runner::run(agent, "Check").await);
        assert!(message.contains("without a valid call_id"), "{message}");
    }
}

#[tokio::test]
async fn process_model_response_rejects_orphan_program_output() {
    // Python parametrizes the output as a model object and as a dict (twin, same JSON here).
    let m = model(vec![vec![program_output("completed")]]);
    let agent = Agent::new("inventory")
        .with_model_instance(m)
        .with_tool(ProgrammaticToolCallingTool);
    let message = behavior_error(Runner::run(agent, "Check").await);
    assert!(
        message.contains("does not match a parent program item"),
        "{message}"
    );
}

#[tokio::test]
async fn process_model_response_accepts_program_output_for_retained_program() {
    // The program is retained from an earlier turn of the run; its output arrives next.
    let m = model(vec![
        vec![program_item()],
        vec![program_output("completed"), text("done")],
    ]);
    let result = Runner::run(
        inventory_agent(m).with_tool(ProgrammaticToolCallingTool),
        "Check",
    )
    .await
    .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert!(raw_types(&result).contains(&"program_output".to_string()));
    assert_eq!(count_outputs(&result), 1);
}

#[tokio::test]
async fn process_model_response_rejects_malformed_program_output() {
    // Python's four cases: status missing, status "running", result missing, result not a string.
    let cases: [(&str, Option<Value>, &str); 4] = [
        ("status", None, "without a valid status"),
        ("status", Some(json!("running")), "without a valid status"),
        ("result", None, "without a string result"),
        ("result", Some(json!(42)), "without a string result"),
    ];
    for (field, value, expected) in cases {
        let mut output = json!({
            "type": "program_output", "id": "program_output_item", "call_id": PROGRAM_CALL_ID,
            "result": "{}", "status": "completed",
        });
        match value {
            Some(v) => output[field] = v,
            None => {
                output.as_object_mut().unwrap().remove(field);
            }
        }
        let m = model(vec![vec![program_item(), OutputItem::from_value(output)]]);
        let agent = Agent::new("inventory")
            .with_model_instance(m)
            .with_tool(ProgrammaticToolCallingTool);
        let message = behavior_error(Runner::run(agent, "Check").await);
        assert!(message.contains(expected), "{field}: {message}");
    }
}

#[tokio::test]
async fn process_model_response_rejects_duplicate_completed_program_output() {
    // Python: a second completed output is rejected whether the first completed in an earlier
    // turn (existing_items) or in the same response.
    let duplicate = json!({
        "type": "program_output", "id": "duplicate_program_output_item",
        "call_id": PROGRAM_CALL_ID, "result": "{}", "status": "completed",
    });
    // Parent completed in an earlier turn.
    let m = model(vec![
        vec![program_item(), program_output("completed")],
        vec![OutputItem::from_value(duplicate.clone())],
    ]);
    let agent = Agent::new("inventory")
        .with_model_instance(m)
        .with_tool(ProgrammaticToolCallingTool);
    let message = behavior_error(Runner::run(agent, "Check").await);
    assert!(
        message.contains("parent program is already completed"),
        "{message}"
    );

    // Parent and both outputs in the same response.
    let m = model(vec![vec![
        program_item(),
        program_output("completed"),
        OutputItem::from_value(duplicate),
    ]]);
    let agent = Agent::new("inventory")
        .with_model_instance(m)
        .with_tool(ProgrammaticToolCallingTool);
    let message = behavior_error(Runner::run(agent, "Check").await);
    assert!(
        message.contains("parent program is already completed"),
        "{message}"
    );
}

#[tokio::test]
async fn process_model_response_rejects_program_owned_call_for_completed_retained_program() {
    let m = model(vec![
        vec![program_item(), program_output("completed")],
        vec![lookup_call()],
    ]);
    let message =
        behavior_error(Runner::run(with_programmatic(m, Some(&["programmatic"])), "Check").await);
    assert!(
        message.contains("parent program is already completed"),
        "{message}"
    );
}

#[tokio::test]
async fn process_model_response_rejects_program_owned_call_when_response_completes_program() {
    // Python: the call and the completed output share one response, in either order.
    for call_before_completed_output in [false, true] {
        let items = if call_before_completed_output {
            vec![program_item(), lookup_call(), program_output("completed")]
        } else {
            vec![program_item(), program_output("completed"), lookup_call()]
        };
        let m = model(vec![items]);
        let message = behavior_error(
            Runner::run(with_programmatic(m, Some(&["programmatic"])), "Check").await,
        );
        assert!(
            message.contains("parent program is already completed"),
            "{message}"
        );
    }
}

#[tokio::test]
async fn process_model_response_accepts_program_owned_call_for_incomplete_retained_program() {
    let m = model(vec![
        vec![program_item(), program_output("incomplete")],
        vec![lookup_call()],
        vec![program_output("completed"), text("done")],
    ]);
    let result = Runner::run(with_programmatic(m, Some(&["programmatic"])), "Check")
        .await
        .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert!(raw_types(&result).contains(&"function_call".to_string()));
}

#[tokio::test]
async fn process_model_response_rejects_program_items_without_programmatic_tool() {
    // Python's four cases: program and program_output, as model objects and dicts (twins).
    for output in [program_item(), program_output("completed")] {
        let m = model(vec![vec![output]]);
        let message = behavior_error(Runner::run(inventory_agent(m), "Check").await);
        assert!(
            message.contains("programmatic_tool_calling tool"),
            "{message}"
        );
    }
}

#[tokio::test]
async fn runner_rejects_program_item_without_programmatic_tool() {
    let m = model(vec![vec![program_item()]]);
    let message = behavior_error(Runner::run(inventory_agent(m), "Check").await);
    assert!(
        message.contains("programmatic_tool_calling tool"),
        "{message}"
    );
}

#[tokio::test]
async fn process_model_response_rejects_program_owned_calls_without_programmatic_tool() {
    // Function, shell and custom calls owned by a program, with no programmatic tool present.
    let calls = program_owned_calls();
    for call in calls {
        let m = model(vec![vec![call]]);
        let agent = inventory_agent(m)
            .with_tool(lookup_tool(Some(&["programmatic"])))
            .with_tool(program_owned_shell())
            .with_tool(program_owned_custom());
        let message = behavior_error(Runner::run(agent, "Check").await);
        assert!(
            message.contains("programmatic_tool_calling tool"),
            "{message}"
        );
    }
}

#[tokio::test]
async fn runner_does_not_execute_program_owned_call_without_programmatic_tool() {
    let m = model(vec![vec![lookup_call()]]);
    let executed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = executed.clone();
    let tool = FunctionTool::new(
        "lookup_inventory",
        "",
        move |_ctx: ToolContext<()>, args: SkuArgs| {
            let flag = flag.clone();
            async move {
                flag.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok::<_, AgentsError>(json!({"sku": args.sku}))
            }
        },
    )
    .with_allowed_callers(vec!["programmatic".into()]);
    let message = behavior_error(Runner::run(inventory_agent(m).with_tool(tool), "Check").await);
    assert!(
        message.contains("programmatic_tool_calling tool"),
        "{message}"
    );
    assert!(!executed.load(std::sync::atomic::Ordering::SeqCst));
}

#[tokio::test]
async fn process_model_response_rejects_program_owned_calls_without_parent_program() {
    // Same six calls as above, with the programmatic tool present but no parent program item.
    for call in program_owned_calls() {
        let m = model(vec![vec![call]]);
        let agent = inventory_agent(m)
            .with_tool(ProgrammaticToolCallingTool)
            .with_tool(lookup_tool(Some(&["programmatic"])))
            .with_tool(program_owned_shell())
            .with_tool(program_owned_custom());
        let message = behavior_error(Runner::run(agent, "Check").await);
        assert!(message.contains("parent program item"), "{message}");
    }
}

#[tokio::test]
async fn process_model_response_accepts_program_owned_call_with_parent_program() {
    // The parent is in the same response, or retained from an earlier turn.
    let m = model(vec![
        vec![program_item(), lookup_call()],
        vec![program_output("completed"), text("done")],
    ]);
    let result = Runner::run(with_programmatic(m, Some(&["programmatic"])), "Check")
        .await
        .unwrap();
    assert!(raw_types(&result).contains(&"function_call".to_string()));

    let m = model(vec![
        vec![program_item()],
        vec![lookup_call()],
        vec![program_output("completed"), text("done")],
    ]);
    let result = Runner::run(with_programmatic(m, Some(&["programmatic"])), "Check")
        .await
        .unwrap();
    assert!(raw_types(&result).contains(&"function_call".to_string()));
}

#[tokio::test]
async fn process_model_response_rejects_program_child_before_parent() {
    // A child listed before its parent in one response is rejected, for both child types.
    for child in [lookup_call(), program_output("completed")] {
        let m = model(vec![vec![child, program_item()]]);
        let message = behavior_error(
            Runner::run(with_programmatic(m, Some(&["programmatic"])), "Check").await,
        );
        assert!(message.contains("parent program item"), "{message}");
    }
}

#[tokio::test]
async fn process_model_response_accepts_server_owned_parent_from_submitted_delta() {
    // Server-managed conversation: the parent program lives on the server, and the submitted
    // input delta evidences it through a function output that names the program caller.
    for child in [lookup_call(), program_output("completed")] {
        let m = model(vec![vec![child.clone()], vec![text("done")]]);
        let delta = vec![json!({
            "type": "function_call_output", "call_id": FUNCTION_CALL_ID,
            "output": "{\"sku\":\"A-1\",\"available_units\":42}", "caller": program_caller(),
        })];
        let result = Runner::run_with_options(
            with_programmatic(m, Some(&["programmatic"])),
            openai_agents_sdk::RunInput::from(delta),
            openai_agents_sdk::RunOptions::new(()).previous_response_id("resp-parent"),
        )
        .await;
        match result {
            Ok(result) => assert_eq!(result.final_output_text().as_deref(), Some("done")),
            Err(err) => panic!("server-owned child rejected: {err}"),
        }
    }
}

#[tokio::test]
async fn process_model_response_rejects_server_owned_child_without_parent_evidence() {
    let m = model(vec![vec![lookup_call()]]);
    let result = Runner::run_with_options(
        with_programmatic(m, Some(&["programmatic"])),
        openai_agents_sdk::RunInput::from("Check"),
        openai_agents_sdk::RunOptions::new(()).previous_response_id("resp-parent"),
    )
    .await;
    let message = behavior_error(result);
    assert!(message.contains("parent program item"), "{message}");
}

#[tokio::test]
async fn process_model_response_accepts_server_owned_parent_after_incomplete_output() {
    // Python keeps the incomplete output in history (`existing_items`); a server-managed run
    // submits it as input. The call that follows is then owned by a server-side parent.
    let m = model(vec![vec![lookup_call()], vec![text("done")]]);
    let submitted = vec![program_output("incomplete").to_value()];
    let result = Runner::run_with_options(
        with_programmatic(m, Some(&["programmatic"])),
        openai_agents_sdk::RunInput::from(submitted),
        openai_agents_sdk::RunOptions::new(()).previous_response_id("resp-parent"),
    )
    .await
    .unwrap();
    assert!(raw_types(&result).contains(&"function_call".to_string()));
}

#[tokio::test]
async fn runner_does_not_execute_program_owned_call_without_parent_program() {
    let m = model(vec![vec![lookup_call()]]);
    let executed = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let flag = executed.clone();
    let tool = FunctionTool::new(
        "lookup_inventory",
        "",
        move |_ctx: ToolContext<()>, args: SkuArgs| {
            let flag = flag.clone();
            async move {
                flag.store(true, std::sync::atomic::Ordering::SeqCst);
                Ok::<_, AgentsError>(json!({"sku": args.sku}))
            }
        },
    )
    .with_allowed_callers(vec!["programmatic".into()]);
    let agent = inventory_agent(m)
        .with_tool(ProgrammaticToolCallingTool)
        .with_tool(tool);
    let message = behavior_error(Runner::run(agent, "Check").await);
    assert!(message.contains("parent program item"), "{message}");
    assert!(!executed.load(std::sync::atomic::Ordering::SeqCst));
}

// ---------------------------------------------------------------------------------------------
// Caller permissions
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn process_model_response_rejects_disallowed_function_callers() {
    // Python's three cases: direct-only tool called by a program; programmatic-only tool with
    // a program caller allowed but the call's caller omitted (direct).
    // Case 1: default callers (direct only), called from a program.
    let m = model(vec![vec![program_item(), lookup_call()]]);
    let message = behavior_error(Runner::run(with_programmatic(m, None), "Check").await);
    assert!(message.contains("caller programmatic"), "{message}");

    // Case 2: explicit direct only, called from a program.
    let m = model(vec![vec![program_item(), lookup_call()]]);
    let message =
        behavior_error(Runner::run(with_programmatic(m, Some(&["direct"])), "Check").await);
    assert!(message.contains("caller programmatic"), "{message}");

    // Case 3: programmatic only, call without any caller (direct).
    let direct_call = OutputItem::from_value(json!({
        "type": "function_call", "id": "function_item", "call_id": FUNCTION_CALL_ID,
        "name": "lookup_inventory", "arguments": "{\"sku\":\"A-1\"}",
    }));
    let m = model(vec![vec![direct_call]]);
    let message =
        behavior_error(Runner::run(with_programmatic(m, Some(&["programmatic"])), "Check").await);
    assert!(message.contains("caller direct"), "{message}");
}

#[tokio::test]
async fn process_model_response_rejects_unknown_function_caller_type() {
    let call = OutputItem::from_value(json!({
        "type": "function_call", "id": "function_item", "call_id": FUNCTION_CALL_ID,
        "name": "lookup_inventory", "arguments": "{\"sku\":\"A-1\"}",
        "caller": {"type": "unknown", "caller_id": PROGRAM_CALL_ID},
    }));
    let m = model(vec![vec![call]]);
    let agent = inventory_agent(m).with_tool(lookup_tool(None));
    let message = behavior_error(Runner::run(agent, "Check").await);
    assert!(
        message.contains("unsupported caller type 'unknown'"),
        "{message}"
    );
}

#[tokio::test]
async fn process_model_response_rejects_disallowed_non_function_callers() {
    // Shell and custom calls from a program are refused by their default (direct) callers, and
    // an apply-patch call without a caller is refused by its programmatic-only allowance.
    let m = model(vec![vec![program_item(), program_owned_shell_call()]]);
    let agent = inventory_agent(m)
        .with_tool(ProgrammaticToolCallingTool)
        .with_tool(ShellTool::new(|_req| async {
            Ok::<_, AgentsError>("ok".to_string())
        }));
    let message = behavior_error(Runner::run(agent, "Check").await);
    assert!(message.contains("caller programmatic"), "{message}");

    let m = model(vec![vec![program_item(), program_owned_custom_call()]]);
    let agent = inventory_agent(m)
        .with_tool(ProgrammaticToolCallingTool)
        .with_tool(CustomTool::new(
            "custom",
            "Custom tool",
            |_ctx: ToolContext<()>, _input: String| async {
                Ok::<_, AgentsError>("ok".to_string())
            },
        ));
    let message = behavior_error(Runner::run(agent, "Check").await);
    assert!(message.contains("caller programmatic"), "{message}");

    let m = model(vec![vec![apply_patch_call_without_caller()]]);
    let agent = inventory_agent(m)
        .with_tool(ProgrammaticToolCallingTool)
        .with_tool(apply_patch_tool(Some(&["programmatic"])));
    let message = behavior_error(Runner::run(agent, "Check").await);
    assert!(message.contains("caller direct"), "{message}");
}

#[tokio::test]
async fn process_model_response_rejects_program_owned_direct_only_hosted_calls() {
    // Python's three hosted calls: file search, web search and image generation, owned by a
    // program while the hosted tool is not declared as programmatic (twins: same JSON).
    let hosted = [
        json!({"id": "file_search_1", "queries": ["inventory"], "status": "completed",
               "type": "file_search_call", "caller": program_caller()}),
        json!({"id": "web_search_1", "action": {"type": "search", "query": "inventory"},
               "status": "completed", "type": "web_search_call", "caller": program_caller()}),
        json!({"id": "image_generation_1", "status": "completed",
               "type": "image_generation_call", "caller": program_caller()}),
    ];
    for call in hosted {
        let m = model(vec![vec![program_item(), OutputItem::from_value(call)]]);
        let agent = inventory_agent(m).with_tool(ProgrammaticToolCallingTool);
        let message = behavior_error(Runner::run(agent, "Check").await);
        assert!(message.contains("caller programmatic"), "{message}");
    }
}

#[tokio::test]
async fn process_model_response_rejects_program_owned_tool_search_items() {
    for item in [
        json!({"id": "tool_search_call_item", "arguments": {}, "execution": "server",
               "status": "completed", "type": "tool_search_call", "caller": program_caller()}),
        json!({"id": "tool_search_output_item", "execution": "server", "status": "completed",
               "tools": [], "type": "tool_search_output", "caller": program_caller()}),
    ] {
        let m = model(vec![vec![program_item(), OutputItem::from_value(item)]]);
        let agent = inventory_agent(m)
            .with_tool(ProgrammaticToolCallingTool)
            .with_tool(ToolSearchTool::new());
        let message = behavior_error(Runner::run(agent, "Check").await);
        assert!(message.contains("caller programmatic"), "{message}");
    }
}

#[tokio::test]
async fn process_model_response_rejects_disallowed_hosted_program_callers() {
    // Hosted MCP approval request, MCP call, MCP tool listing and code interpreter, each with
    // the allowance omitted (direct only) and explicitly direct: eight Python cases.
    for allowed in [None, Some(vec!["direct"])] {
        for output in hosted_program_outputs() {
            let (item, tool) = hosted_program_call_and_tool(output, allowed.clone());
            let m = model(vec![vec![program_item(), item]]);
            let agent = inventory_agent(m)
                .with_tool(ProgrammaticToolCallingTool)
                .with_tool(tool);
            let message = behavior_error(Runner::run(agent, "Check").await);
            assert!(
                message.contains("caller programmatic"),
                "{output}: {message}"
            );
        }
    }
}

#[tokio::test]
async fn process_model_response_accepts_allowed_hosted_program_callers() {
    // With the programmatic caller allowed, the hosted items are accepted (no error).
    for output in hosted_program_outputs() {
        let (item, tool) = hosted_program_call_and_tool(output, Some(vec!["programmatic"]));
        let m = model(vec![vec![program_item(), item], vec![text("done")]]);
        let agent = inventory_agent(m)
            .with_tool(ProgrammaticToolCallingTool)
            .with_tool(tool);
        let result = Runner::run(agent, "Check").await;
        match result {
            // An MCP approval request pauses the run for approval instead of completing.
            Ok(result) if output == "mcp_approval_request" => {
                assert_eq!(result.interruptions.len(), 1, "{output}");
            }
            Ok(result) => assert_eq!(
                result.final_output_text().as_deref(),
                Some("done"),
                "{output}"
            ),
            Err(err) => panic!("{output}: {err}"),
        }
    }
}

#[tokio::test]
async fn process_model_response_rejects_disallowed_program_owned_shell_output() {
    // Shell output owned by a program is refused unless the shell tool allows programmatic.
    for allowed in [None, Some(vec!["direct"])] {
        let m = model(vec![vec![program_item(), program_owned_shell_output()]]);
        let shell = shell_tool(allowed.as_deref());
        let agent = inventory_agent(m)
            .with_tool(ProgrammaticToolCallingTool)
            .with_tool(shell);
        let message = behavior_error(Runner::run(agent, "Check").await);
        assert!(message.contains("caller programmatic"), "{message}");
    }
}

#[tokio::test]
async fn process_model_response_accepts_allowed_program_owned_shell_output() {
    let m = model(vec![
        vec![program_item(), program_owned_shell_output()],
        vec![text("done")],
    ]);
    let agent = inventory_agent(m)
        .with_tool(ProgrammaticToolCallingTool)
        .with_tool(shell_tool(Some(&["programmatic"])));
    let result = Runner::run(agent, "Check").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
}

// ---------------------------------------------------------------------------------------------
// Fixtures for the caller-permission cases
// ---------------------------------------------------------------------------------------------

fn shell_tool(callers: Option<&[&str]>) -> ShellTool<()> {
    let tool = ShellTool::new(|_req| async { Ok::<_, AgentsError>("ok".to_string()) });
    match callers {
        Some(c) => tool.with_allowed_callers(c.iter().map(|s| s.to_string()).collect()),
        None => tool,
    }
}

fn program_owned_shell() -> ShellTool<()> {
    shell_tool(Some(&["programmatic"]))
}

fn program_owned_custom() -> CustomTool<()> {
    CustomTool::new(
        "custom",
        "Custom tool",
        |_ctx: ToolContext<()>, _input: String| async { Ok::<_, AgentsError>("ok".to_string()) },
    )
    .with_allowed_callers(vec!["programmatic".into()])
}

fn apply_patch_tool(callers: Option<&[&str]>) -> ApplyPatchTool<()> {
    let tool = ApplyPatchTool::new(FakeEditor);
    match callers {
        Some(c) => tool.with_allowed_callers(c.iter().map(|s| s.to_string()).collect()),
        None => tool,
    }
}

struct FakeEditor;

#[async_trait::async_trait]
impl openai_agents_sdk::editor::ApplyPatchEditor for FakeEditor {
    async fn create_file(
        &self,
        _operation: &ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<ApplyPatchResult> {
        Ok(ApplyPatchResult::default())
    }
    async fn update_file(
        &self,
        _operation: &ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<ApplyPatchResult> {
        Ok(ApplyPatchResult::default())
    }
    async fn delete_file(
        &self,
        _operation: &ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<ApplyPatchResult> {
        Ok(ApplyPatchResult::default())
    }
}

fn program_owned_calls() -> Vec<OutputItem> {
    vec![
        lookup_call(),
        program_owned_shell_call(),
        program_owned_custom_call(),
    ]
}

fn program_owned_shell_call() -> OutputItem {
    OutputItem::from_value(json!({
        "id": "shell_item", "call_id": "call_shell", "action": {"commands": ["echo ok"]},
        "status": "completed", "type": "shell_call", "caller": program_caller(),
    }))
}

fn program_owned_custom_call() -> OutputItem {
    OutputItem::from_value(json!({
        "id": "custom_item", "call_id": "call_custom", "input": "input", "name": "custom",
        "type": "custom_tool_call", "caller": program_caller(),
    }))
}

fn apply_patch_call_without_caller() -> OutputItem {
    OutputItem::from_value(json!({
        "id": "apply_patch_item", "call_id": "call_apply_patch",
        "operation": {"type": "create_file", "path": "example.txt", "diff": "hello"},
        "status": "completed", "type": "apply_patch_call",
    }))
}

fn program_owned_shell_output() -> OutputItem {
    OutputItem::from_value(json!({
        "id": "shell_output_1", "call_id": "shell_call_1", "status": "completed",
        "type": "shell_call_output", "output": [], "caller": program_caller(),
    }))
}

/// `(type, extra fields)` for the hosted programmatic outputs Python parametrizes.
fn hosted_program_outputs() -> Vec<&'static str> {
    vec![
        "mcp_approval_request",
        "mcp_call",
        "mcp_list_tools",
        "code_interpreter_call",
    ]
}

/// The hosted call item and the hosted tool that owns its `allowed_callers`.
fn hosted_program_call_and_tool(
    output_type: &str,
    allowed_callers: Option<Vec<&str>>,
) -> (OutputItem, Tool) {
    let callers = allowed_callers.map(|c| json!(c));
    if output_type == "code_interpreter_call" {
        let mut config = json!({"type": "code_interpreter", "container": "auto"});
        if let Some(c) = &callers {
            config["allowed_callers"] = c.clone();
        }
        let item = OutputItem::from_value(json!({
            "id": "code_interpreter_1", "container_id": "container_1",
            "status": "completed", "type": "code_interpreter_call", "caller": program_caller(),
        }));
        return (
            item,
            Tool::CodeInterpreter(CodeInterpreterTool {
                tool_config: config,
            }),
        );
    }
    let mut config = json!({
        "type": "mcp", "server_label": "docs_server", "server_url": "https://example.com/mcp",
    });
    if let Some(c) = &callers {
        config["allowed_callers"] = c.clone();
    }
    let item = match output_type {
        "mcp_approval_request" => json!({
            "id": "mcp_approval_request_1", "arguments": "{}", "name": "search_docs",
            "server_label": "docs_server", "type": "mcp_approval_request", "caller": program_caller(),
        }),
        "mcp_call" => json!({
            "id": "mcp_call_1", "arguments": "{}", "name": "search_docs",
            "server_label": "docs_server", "type": "mcp_call", "caller": program_caller(),
        }),
        _ => json!({
            "id": "mcp_list_tools_1", "server_label": "docs_server", "tools": [],
            "type": "mcp_list_tools", "caller": program_caller(),
        }),
    };
    (
        OutputItem::from_value(item),
        Tool::HostedMCP(HostedMCPTool::new(config)),
    )
}

type Tool = openai_agents_sdk::tool::Tool<()>;
