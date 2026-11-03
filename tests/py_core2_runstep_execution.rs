//! Ported from the Python SDK's `tests/test_run_step_execution.py`.
//!
//! Python drives `run_loop.process_model_response` + `execute_tools_and_side_effects` for one
//! response and inspects the `SingleStepResult`. Rust's step machinery is private, so each port
//! runs the same response through `Runner` and reads the same decision off the run:
//! - `NextStepFinalOutput(x)`: the run ends with final output `x` after one model call;
//! - `NextStepRunAgain`: the model is called a second time (scripted to finish);
//! - `NextStepHandoff(agent)`: `last_agent` is the target;
//! - `NextStepInterruption`: the result carries the pending approvals;
//! - `generated_items`: the items of that one turn in `new_items`.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use async_trait::async_trait;
use openai_agents_sdk::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchResult};
use openai_agents_sdk::handoffs::default_tool_name;
use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::run_config::ToolExecutionConfig;
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full, refusal_message,
};
use openai_agents_sdk::tool::{
    ApplyPatchTool, ApprovalDecision, FunctionTool, MCPToolApprovalResult, ShellTool,
    ToolTimeoutBehavior,
};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::tool_guardrails::{ToolGuardrailFunctionOutput, ToolInputGuardrail};
use openai_agents_sdk::tracing::{Span, Trace, TracingProcessor, set_trace_processors};
use openai_agents_sdk::{
    Agent, AgentsError, HostedMCPTool, InputItem, RunConfig, RunItem, RunOptions, RunResult, Runner,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

type Log = Arc<Mutex<Vec<String>>>;

/// Records every finished span, replacing the default (network) exporter for this binary.
#[derive(Debug, Default)]
struct Recorder {
    ends: Mutex<Vec<Value>>,
}

impl TracingProcessor for Recorder {
    fn on_trace_start(&self, _trace: &Trace) {}
    fn on_trace_end(&self, _trace: &Trace) {}
    fn on_span_start(&self, _span: &Span) {}
    fn on_span_end(&self, span: &Span) {
        if let Some(export) = span.export() {
            self.ends.lock().unwrap().push(export);
        }
    }
}

fn recorder() -> Arc<Recorder> {
    static RECORDER: OnceLock<Arc<Recorder>> = OnceLock::new();
    RECORDER
        .get_or_init(|| {
            openai_agents_sdk::tracing::set_tracing_disabled(false);
            let recorder = Arc::new(Recorder::default());
            set_trace_processors(vec![recorder.clone() as Arc<dyn TracingProcessor>]);
            recorder
        })
        .clone()
}

/// Function spans recorded for `tool_name` (each test uses its own tool names).
fn function_spans(tool_name: &str) -> Vec<Value> {
    recorder()
        .ends
        .lock()
        .unwrap()
        .iter()
        .filter(|s| s["span_data"]["type"] == "function" && s["span_data"]["name"] == tool_name)
        .cloned()
        .collect()
}

fn scripted(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    let model = Arc::new(ScriptedModel::new());
    for step in steps {
        model.enqueue(ScriptedStep::output(step)).unwrap();
    }
    model
}

fn call(name: &str, args: &str, call_id: &str) -> OutputItem {
    function_call_full(name, args, call_id, None, None)
}

fn get_function_tool(name: &str, value: &str) -> FunctionTool<()> {
    let value = value.to_string();
    FunctionTool::new(name, "", move |_c, _a: Value| {
        let value = value.clone();
        async move { Ok(value) }
    })
}

async fn run_with(
    agent: Agent<()>,
    input: impl Into<openai_agents_sdk::RunInput>,
    config: RunConfig<()>,
) -> openai_agents_sdk::Result<RunResult<()>> {
    recorder();
    Runner::run_with_options(&agent, input, RunOptions::new(()).run_config(config)).await
}

async fn run(agent: Agent<()>) -> openai_agents_sdk::Result<RunResult<()>> {
    run_with(agent, "hello", RunConfig::default()).await
}

fn assert_message(item: &RunItem, text: &str) {
    match &item.kind {
        RunItemKind::MessageOutput { raw_item } => {
            assert_eq!(raw_item.role, "assistant");
            assert_eq!(raw_item.text_content(), text);
        }
        other => panic!("expected message, got {other:?}"),
    }
}

fn assert_function_call(item: &RunItem, name: &str, arguments: Option<&str>) {
    match &item.kind {
        RunItemKind::ToolCall { raw_item } => {
            assert_eq!(raw_item["type"], "function_call");
            assert_eq!(raw_item["name"], name);
            if let Some(arguments) = arguments {
                assert_eq!(raw_item["arguments"], arguments);
            }
        }
        other => panic!("expected tool call, got {other:?}"),
    }
}

fn assert_function_output(item: &RunItem, output: &str) {
    match &item.kind {
        RunItemKind::ToolCallOutput { raw_item, .. } => {
            assert_eq!(raw_item["type"], "function_call_output");
            assert_eq!(raw_item["output"], output);
        }
        other => panic!("expected tool output, got {other:?}"),
    }
}

/// The items generated by the first turn: everything before the second turn's final message.
fn first_turn(result: &RunResult<()>) -> &[RunItem] {
    &result.new_items[..result.new_items.len() - 1]
}

fn assert_user_error(err: AgentsError, needle: &str) {
    match err {
        AgentsError::UserError { message, .. } => {
            assert!(message.contains(needle), "{message:?} lacks {needle:?}")
        }
        other => panic!("expected UserError, got {other:?}"),
    }
}

fn assert_model_behavior(err: AgentsError, needle: &str) {
    match err {
        AgentsError::ModelBehaviorError { message, .. } => {
            assert!(message.contains(needle), "{message:?} lacks {needle:?}")
        }
        other => panic!("expected ModelBehaviorError, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------------------------
// Final output / run again
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn empty_response_is_final_output() {
    let model = scripted(vec![vec![]]);
    let result = run(Agent::new("test").with_model_instance(model.clone()))
        .await
        .unwrap();
    assert_eq!(result.input, vec![InputItem::user("hello")]);
    assert!(result.new_items.is_empty());
    assert_eq!(result.final_output, Some(json!("")));
    assert_eq!(model.calls().len(), 1);
}

#[tokio::test]
async fn plaintext_agent_no_tool_calls_is_final_output() {
    let result = run(Agent::new("test")
        .with_model_instance(scripted(vec![vec![assistant_message("hello_world")]])))
    .await
    .unwrap();
    assert_eq!(result.new_items.len(), 1);
    assert_message(&result.new_items[0], "hello_world");
    assert_eq!(result.final_output_text().as_deref(), Some("hello_world"));
}

#[tokio::test]
async fn plaintext_agent_no_tool_calls_multiple_messages_is_final_output() {
    let agent = Agent::new("test").with_model_instance(scripted(vec![vec![
        assistant_message("hello_world"),
        assistant_message("bye"),
    ]]));
    let result = run_with(
        agent,
        vec![InputItem::user("test"), InputItem::user("test2")],
        RunConfig::default(),
    )
    .await
    .unwrap();
    assert_eq!(result.input.len(), 2);
    assert_eq!(result.new_items.len(), 2);
    assert_message(&result.new_items[0], "hello_world");
    assert_message(&result.new_items[1], "bye");
    assert_eq!(result.final_output_text().as_deref(), Some("bye"));
}

#[tokio::test]
async fn plaintext_agent_with_tool_call_is_run_again() {
    let model = scripted(vec![
        vec![assistant_message("hello_world"), call("test", "", "c1")],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::new("test")
        .with_model_instance(model.clone())
        .with_tool(get_function_tool("test", "123"));
    let result = run(agent).await.unwrap();
    let items = first_turn(&result);
    assert_eq!(items.len(), 3);
    assert_message(&items[0], "hello_world");
    assert_function_call(&items[1], "test", None);
    assert_function_output(&items[2], "123");
    assert_eq!(model.calls().len(), 2, "run again");
}

fn tracked_tool(active: Arc<AtomicUsize>, max_seen: Arc<AtomicUsize>) -> FunctionTool<()> {
    #[derive(Deserialize, JsonSchema)]
    struct Args {
        value: i64,
    }
    FunctionTool::new("tracked_tool", "", move |_c, a: Args| {
        let active = active.clone();
        let max_seen = max_seen.clone();
        async move {
            let now = active.fetch_add(1, Ordering::SeqCst) + 1;
            max_seen.fetch_max(now, Ordering::SeqCst);
            let delay = if a.value == 1 { 30 } else { 5 };
            tokio::time::sleep(Duration::from_millis(delay)).await;
            active.fetch_sub(1, Ordering::SeqCst);
            Ok(format!("ok-{}", a.value))
        }
    })
}

async fn concurrency_case(cap: Option<usize>) -> (usize, Vec<RunItem>) {
    let active = Arc::new(AtomicUsize::new(0));
    let max_seen = Arc::new(AtomicUsize::new(0));
    let agent = Agent::new("test")
        .with_model_instance(scripted(vec![
            (1..=3)
                .map(|i| {
                    call(
                        "tracked_tool",
                        &json!({"value": i}).to_string(),
                        &format!("call_{i}"),
                    )
                })
                .collect(),
            vec![assistant_message("done")],
        ]))
        .with_tool(tracked_tool(active.clone(), max_seen.clone()));
    let mut config = RunConfig::default();
    if let Some(cap) = cap {
        config = config.with_tool_execution(ToolExecutionConfig {
            max_function_tool_concurrency: Some(cap),
            ..Default::default()
        });
    }
    let result = run_with(agent, "hello", config).await.unwrap();
    assert_eq!(active.load(Ordering::SeqCst), 0);
    (max_seen.load(Ordering::SeqCst), result.new_items)
}

#[tokio::test]
async fn function_tool_concurrency_default_starts_all_calls() {
    let (max_seen, items) = concurrency_case(None).await;
    assert_eq!(max_seen, 3);
    assert_function_output(&items[3], "ok-1");
    assert_function_output(&items[4], "ok-2");
    assert_function_output(&items[5], "ok-3");
}

#[tokio::test]
async fn function_tool_concurrency_cap_limits_calls_and_preserves_output_order() {
    let (max_seen, items) = concurrency_case(Some(2)).await;
    assert_eq!(max_seen, 2);
    assert_function_output(&items[3], "ok-1");
    assert_function_output(&items[4], "ok-2");
    assert_function_output(&items[5], "ok-3");
}

fn logging_tool(name: &str, log: &Log, result: Result<&str, &str>) -> FunctionTool<()> {
    let log = log.clone();
    let tool_name = name.to_string();
    let result = result.map(str::to_string).map_err(str::to_string);
    FunctionTool::new(name, "", move |_c, _a: Value| {
        log.lock().unwrap().push(tool_name.clone());
        let result = result.clone();
        async move { result.map_err(AgentsError::other) }
    })
}

#[tokio::test]
async fn function_tool_concurrency_cap_leaves_queued_calls_unstarted_after_failure() {
    let log = Log::default();
    let agent = Agent::new("test")
        .with_model_instance(scripted(vec![vec![
            call("failing_tool", "{}", "call_1"),
            call("queued_tool", "{}", "call_2"),
        ]]))
        .with_tool(logging_tool("failing_tool", &log, Err("boom")).without_failure_error_function())
        .with_tool(logging_tool("queued_tool", &log, Ok("should-not-run")));
    let config = RunConfig::default().with_tool_execution(ToolExecutionConfig {
        max_function_tool_concurrency: Some(1),
        ..Default::default()
    });
    let err = run_with(agent, "hello", config).await.unwrap_err();
    assert_user_error(err, "Error running tool failing_tool: boom");
    assert_eq!(*log.lock().unwrap(), vec!["failing_tool"]);
}

fn hosted_shell() -> ShellTool<()> {
    ShellTool::hosted(json!({"type": "container_auto"})).unwrap()
}

fn shell_call(call_id: &str, id: &str) -> OutputItem {
    OutputItem::from_value(json!({"type": "shell_call", "id": id, "call_id": call_id,
        "status": "completed", "action": {"commands": ["echo hi"], "timeout_ms": 1000}}))
}

fn shell_output(call_id: &str, id: &str) -> OutputItem {
    OutputItem::from_value(
        json!({"type": "shell_call_output", "id": id, "call_id": call_id,
        "status": "completed",
        "output": [{"stdout": "hi\n", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}}]}),
    )
}

#[tokio::test]
async fn plaintext_agent_hosted_shell_items_without_message_runs_again() {
    let model = scripted(vec![
        vec![
            shell_call("call_shell_hosted", "shell_call_hosted"),
            shell_output("call_shell_hosted", "sh_out_hosted"),
        ],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::new("test")
        .with_model_instance(model.clone())
        .with_tool(hosted_shell());
    let result = run(agent).await.unwrap();
    let items = first_turn(&result);
    assert_eq!(items.len(), 2);
    assert!(matches!(items[0].kind, RunItemKind::ToolCall { .. }));
    assert!(matches!(items[1].kind, RunItemKind::ToolCallOutput { .. }));
    assert_eq!(model.calls().len(), 2);
}

#[tokio::test]
async fn plaintext_agent_shell_output_only_without_message_runs_again() {
    let model = scripted(vec![
        vec![shell_output("call_shell_only", "sh_out_only")],
        vec![assistant_message("done")],
    ]);
    let result = run(Agent::new("test").with_model_instance(model.clone()))
        .await
        .unwrap();
    let items = first_turn(&result);
    assert_eq!(items.len(), 1);
    assert!(matches!(items[0].kind, RunItemKind::ToolCallOutput { .. }));
    assert_eq!(model.calls().len(), 2);
}

#[tokio::test]
async fn plaintext_agent_tool_search_only_without_message_runs_again() {
    let model = scripted(vec![
        vec![
            OutputItem::from_value(json!({"type": "tool_search_call", "id": "tsc_step",
                "arguments": {"paths": ["crm"], "query": "profile"}, "execution": "server",
                "status": "completed"})),
            OutputItem::from_value(json!({"type": "tool_search_output", "id": "tso_step",
                "execution": "server", "status": "completed",
                "tools": [{"type": "function", "name": "lookup_account",
                           "description": "Look up a CRM account.",
                           "parameters": {"type": "object",
                                          "properties": {"account_id": {"type": "string"}},
                                          "required": ["account_id"]},
                           "defer_loading": true}]})),
        ],
        vec![assistant_message("done")],
    ]);
    let result = run(Agent::new("test").with_model_instance(model.clone()))
        .await
        .unwrap();
    let items = first_turn(&result);
    assert_eq!(items.len(), 2);
    assert_eq!(items[0].to_input_item().0["type"], "tool_search_call");
    assert_eq!(items[1].to_input_item().0["type"], "tool_search_output");
    assert_eq!(model.calls().len(), 2);
}

#[tokio::test]
async fn plaintext_agent_client_tool_search_requires_manual_handling() {
    let agent = Agent::new("test").with_model_instance(scripted(vec![vec![
        OutputItem::from_value(json!({"type": "tool_search_call", "id": "tsc_client_step",
            "call_id": "call_tool_search_client", "arguments": {"paths": ["crm"], "query": "profile"},
            "execution": "client", "status": "completed"})),
    ]]));
    assert_model_behavior(
        run(agent).await.unwrap_err(),
        "Client-executed tool_search calls",
    );
}

#[tokio::test]
async fn plaintext_agent_hosted_shell_with_refusal_message_raises_refusal_error() {
    let agent = Agent::new("test")
        .with_model_instance(scripted(vec![vec![
            shell_call("call_shell_hosted_refusal", "shell_call_hosted_refusal"),
            shell_output("call_shell_hosted_refusal", "sh_out_hosted_refusal"),
            refusal_message("I cannot help with that."),
        ]]))
        .with_tool(hosted_shell());
    match run(agent).await.unwrap_err() {
        AgentsError::ModelRefusalError { refusal, .. } => {
            assert_eq!(refusal, "I cannot help with that.")
        }
        other => panic!("expected refusal, got {other:?}"),
    }
}

// ---------------------------------------------------------------------------------------------
// Multiple tool calls and failures
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn multiple_tool_calls() {
    let model = scripted(vec![
        vec![
            assistant_message("Hello, world!"),
            call("test_1", "{}", "test-1"),
            call("test_2", "{}", "test-2"),
        ],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::new("test")
        .with_model_instance(model.clone())
        .with_tool(get_function_tool("test_1", "123"))
        .with_tool(get_function_tool("test_2", "456"))
        .with_tool(get_function_tool("test_3", "789"));
    let result = run(agent).await.unwrap();
    let items = first_turn(&result);
    assert_eq!(items.len(), 5);
    assert_message(&items[0], "Hello, world!");
    assert_function_call(&items[1], "test_1", None);
    assert_function_call(&items[2], "test_2", None);
    assert_eq!(model.calls().len(), 2);
}

#[tokio::test]
async fn multiple_tool_calls_with_tool_context() {
    #[derive(Deserialize, JsonSchema)]
    struct Args {
        value: String,
    }
    let tool =
        FunctionTool::new(
            "fake_tool",
            "",
            |ctx: ToolContext<()>, a: Args| async move {
                Ok(format!("{}-{}", a.value, ctx.tool_call_id))
            },
        )
        .without_failure_error_function();
    let v1 = json!({"value": "123"}).to_string();
    let v2 = json!({"value": "456"}).to_string();
    let agent = Agent::new("test")
        .with_model_instance(scripted(vec![
            vec![call("fake_tool", &v1, "1"), call("fake_tool", &v2, "2")],
            vec![assistant_message("done")],
        ]))
        .with_tool(tool);
    let result = run(agent).await.unwrap();
    let items = first_turn(&result);
    assert_eq!(items.len(), 4);
    assert_function_call(&items[0], "fake_tool", Some(&v1));
    assert_function_call(&items[1], "fake_tool", Some(&v2));
    assert_function_output(&items[2], "123-1");
    assert_function_output(&items[3], "456-2");
}

#[tokio::test]
async fn multiple_tool_calls_still_raise_when_sibling_failure_error_function_none() {
    let log = Log::default();
    let agent = Agent::new("test")
        .with_model_instance(scripted(vec![vec![
            call("ok_tool", "{}", "1"),
            call("error_tool", "{}", "2"),
        ]]))
        .with_tool(logging_tool("ok_tool", &log, Ok("ok")).without_failure_error_function())
        .with_tool(logging_tool("error_tool", &log, Err("boom")).without_failure_error_function());
    assert_user_error(
        run(agent).await.unwrap_err(),
        "Error running tool error_tool: boom",
    );
}

#[tokio::test]
async fn single_tool_call_still_raises_normal_exception() {
    let log = Log::default();
    let agent = Agent::new("test")
        .with_model_instance(scripted(vec![vec![call("error_tool_solo", "{}", "1")]]))
        .with_tool(
            logging_tool("error_tool_solo", &log, Err("boom")).without_failure_error_function(),
        );
    assert_user_error(
        run(agent).await.unwrap_err(),
        "Error running tool error_tool_solo: boom",
    );
}

#[tokio::test]
async fn multiple_tool_calls_preserve_tool_call_order_for_same_batch_failures() {
    let log = Log::default();
    let agent = Agent::new("test")
        .with_model_instance(scripted(vec![vec![
            call("error_tool_1", "{}", "1"),
            call("error_tool_2", "{}", "2"),
        ]]))
        .with_tool(
            logging_tool("error_tool_1", &log, Err("boom-1")).without_failure_error_function(),
        )
        .with_tool(
            logging_tool("error_tool_2", &log, Err("boom-2")).without_failure_error_function(),
        );
    assert_user_error(
        run(agent).await.unwrap_err(),
        "Error running tool error_tool_1: boom-1",
    );
}

#[tokio::test]
async fn multiple_tool_calls_still_raise_tool_timeout_error() {
    let slow = FunctionTool::new("slow_tool", "", |_c, _a: Value| async move {
        tokio::time::sleep(Duration::from_millis(200)).await;
        Ok("slow")
    })
    .with_timeout(Duration::from_millis(10))
    .with_timeout_behavior(ToolTimeoutBehavior::RaiseException);
    let log = Log::default();
    let agent = Agent::new("test")
        .with_model_instance(scripted(vec![vec![
            call("ok_tool", "{}", "1"),
            call("slow_tool", "{}", "2"),
        ]]))
        .with_tool(logging_tool("ok_tool", &log, Ok("ok")).without_failure_error_function())
        .with_tool(slow);
    let err = run(agent).await.unwrap_err();
    assert!(
        matches!(err, AgentsError::ToolTimeoutError { .. }),
        "{err:?}"
    );
    assert!(err.to_string().contains("timed out"));
}

fn echo_tool(name: &str) -> FunctionTool<()> {
    #[derive(Deserialize, JsonSchema)]
    struct Args {
        value: String,
    }
    FunctionTool::new(name, "", |_c, a: Args| async move { Ok(a.value) })
}

#[tokio::test]
async fn multiple_tool_calls_still_raise_model_behavior_error_when_failure_error_none() {
    let log = Log::default();
    let agent = Agent::new("test")
        .with_model_instance(scripted(vec![vec![
            call("ok_tool", "{}", "1"),
            call("guarded_tool", "bad_json", "2"),
        ]]))
        .with_tool(logging_tool("ok_tool", &log, Ok("ok")).without_failure_error_function())
        .with_tool(echo_tool("guarded_tool").without_failure_error_function());
    assert_model_behavior(
        run(agent).await.unwrap_err(),
        "Invalid JSON input for tool guarded_tool",
    );
}

#[tokio::test]
async fn invalid_json_raises_with_failure_error_function_none() {
    let agent = Agent::new("test")
        .with_model_instance(scripted(vec![vec![call("guarded_solo", "bad_json", "2")]]))
        .with_tool(echo_tool("guarded_solo").without_failure_error_function());
    assert_model_behavior(run(agent).await.unwrap_err(), "Invalid JSON input for tool");
}

#[tokio::test]
async fn input_guardrail_runs_on_invalid_json() {
    let calls = Log::default();
    let calls_in = calls.clone();
    let guardrail = ToolInputGuardrail::new("check", move |data| {
        calls_in
            .lock()
            .unwrap()
            .push(data.context.tool_arguments.clone());
        async move { Ok(ToolGuardrailFunctionOutput::allow().with_output_info(json!("checked"))) }
    });
    let agent = Agent::new("test")
        .with_model_instance(scripted(vec![
            vec![call("guarded", "bad_json", "1")],
            vec![assistant_message("done")],
        ]))
        .with_tool(echo_tool("guarded").with_input_guardrail(guardrail));
    let result = run(agent).await.unwrap();
    assert_eq!(*calls.lock().unwrap(), vec!["bad_json"]);
    assert_eq!(
        result.tool_input_guardrail_results[0].output.output_info,
        json!("checked")
    );
    let output = result
        .new_items
        .iter()
        .find_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { raw_item, .. } => Some(raw_item["output"].clone()),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        output,
        json!("An error occurred while running the tool. Please try again.")
    );
}

#[tokio::test]
async fn function_tool_disabled_before_execution_fails_before_starting_siblings() {
    let checks = Arc::new(Mutex::new(Vec::<bool>::new()));
    let checks_in = checks.clone();
    let log = Log::default();
    let lookup = logging_tool("lookup_secret", &log, Ok("secret")).with_is_enabled_fn(
        move |_ctx, _agent| {
            let mut c = checks_in.lock().unwrap();
            let enabled = c.is_empty();
            c.push(enabled);
            async move { enabled }
        },
    );
    let agent = Agent::new("test")
        .with_model_instance(scripted(vec![vec![
            call("lookup_secret", "{}", "call-1"),
            call("record_side_effect", "{}", "call-2"),
        ]]))
        .with_tool(lookup)
        .with_tool(logging_tool("record_side_effect", &log, Ok("recorded")));
    assert_model_behavior(
        run(agent).await.unwrap_err(),
        "lookup_secret is currently disabled",
    );
    assert_eq!(*checks.lock().unwrap(), vec![true, false]);
    assert!(log.lock().unwrap().is_empty());
}

#[tokio::test]
async fn function_tool_context_includes_run_config() {
    let tool = FunctionTool::new_no_args(
        "tool_with_run_config",
        "",
        |ctx: ToolContext<()>| async move {
            let config = ctx.run_config.clone().expect("run config");
            Ok(match &config.model {
                Some(openai_agents_sdk::AgentModel::Name(n)) => n.clone(),
                _ => "?".into(),
            })
        },
    )
    .without_failure_error_function();
    // The agent's own model answers; the run config only names a model.
    let model = scripted(vec![
        vec![call("tool_with_run_config", "{}", "call-1")],
        vec![assistant_message("done")],
    ]);
    let config = RunConfig::default().with_model("gpt-4.1-mini");
    let agent = Agent::new("test").with_tool(tool);
    let provider_model = model.clone();
    let config = config.with_model_provider(Arc::new(StaticProvider(provider_model)));
    let result = run_with(agent, "hello", config).await.unwrap();
    let items = first_turn(&result);
    assert_eq!(items.len(), 2);
    assert_function_output(&items[1], "gpt-4.1-mini");
}

/// Resolves every model name to one scripted model.
#[derive(Debug)]
struct StaticProvider(Arc<ScriptedModel>);

impl openai_agents_sdk::ModelProvider for StaticProvider {
    fn get_model(
        &self,
        _name: Option<&str>,
    ) -> openai_agents_sdk::Result<Arc<dyn openai_agents_sdk::Model>> {
        Ok(self.0.clone())
    }
}

#[tokio::test]
async fn deferred_function_tool_context_preserves_search_loaded_namespace() {
    let tool = FunctionTool::new_no_args("get_weather", "", |ctx: ToolContext<()>| async move {
        let call_ns = ctx
            .tool_call
            .as_ref()
            .and_then(|c| c.extra.get("namespace").cloned());
        Ok(format!(
            "{{\"tool_call_namespace\": {}, \"tool_namespace\": {}}}",
            call_ns.unwrap_or(Value::Null),
            json!(ctx.tool_namespace)
        ))
    })
    .with_defer_loading(true)
    .without_failure_error_function();
    let agent = Agent::new("test")
        .with_model_instance(scripted(vec![
            vec![function_call_full(
                "get_weather",
                "{}",
                "call-1",
                None,
                Some("get_weather".into()),
            )],
            vec![assistant_message("done")],
        ]))
        .with_tool(tool);
    let result = run(agent).await.unwrap();
    let items = first_turn(&result);
    assert_eq!(items.len(), 2);
    assert_function_output(
        &items[1],
        r#"{"tool_call_namespace": "get_weather", "tool_namespace": "get_weather"}"#,
    );
}

// ---------------------------------------------------------------------------------------------
// Handoffs and structured final output
// ---------------------------------------------------------------------------------------------

fn handoff_call(target: &str) -> OutputItem {
    call(&default_tool_name(target), "{}", &format!("h-{target}"))
}

fn target(name: &str) -> Arc<Agent<()>> {
    Arc::new(Agent::new(name).with_model_instance(scripted(vec![vec![assistant_message(name)]])))
}

#[tokio::test]
async fn handoff_output_leads_to_handoff_next_step() {
    let a1 = target("test_1");
    let agent = Agent::new("test_3")
        .with_model_instance(scripted(vec![vec![
            assistant_message("Hello, world!"),
            handoff_call("test_1"),
        ]]))
        .with_handoff(a1)
        .with_handoff(target("test_2"));
    let result = run(agent).await.unwrap();
    assert_eq!(result.last_agent.name, "test_1");
    // message, handoff call, handoff output; then the target's own message.
    assert_eq!(first_turn(&result).len(), 3);
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct Foo {
    bar: String,
}

#[tokio::test]
async fn final_output_without_tool_runs_again() {
    let model = scripted(vec![
        vec![call("tool_1", "{}", "c1")],
        vec![assistant_message(r#"{"bar":"x"}"#)],
    ]);
    let agent = Agent::new("test")
        .with_model_instance(model.clone())
        .with_output_type::<Foo>()
        .with_tool(get_function_tool("tool_1", "result"));
    let result = run(agent).await.unwrap();
    assert_eq!(first_turn(&result).len(), 2, "tool call, tool call output");
    assert_eq!(model.calls().len(), 2);
}

#[tokio::test]
async fn final_output_leads_to_final_output_next_step() {
    let agent = Agent::new("test")
        .with_model_instance(scripted(vec![vec![
            assistant_message("Hello, world!"),
            assistant_message(r#"{"bar":"123"}"#),
        ]]))
        .with_output_type::<Foo>();
    let result = run(agent).await.unwrap();
    assert_eq!(
        result.final_output_as::<Foo>().unwrap(),
        Foo { bar: "123".into() }
    );
}

#[tokio::test]
async fn handoff_and_final_output_leads_to_handoff_next_step() {
    let agent = Agent::new("test_3")
        .with_model_instance(scripted(vec![vec![
            assistant_message(r#"{"bar":"123"}"#),
            handoff_call("test_1"),
        ]]))
        .with_output_type::<Foo>()
        .with_handoff(target("test_1"))
        .with_handoff(target("test_2"));
    let result = run(agent).await.unwrap();
    assert_eq!(result.last_agent.name, "test_1");
}

#[tokio::test]
async fn multiple_final_output_leads_to_final_output_next_step() {
    let agent = Agent::new("test_3")
        .with_model_instance(scripted(vec![vec![
            assistant_message(r#"{"bar":"123"}"#),
            assistant_message(r#"{"bar":"456"}"#),
        ]]))
        .with_output_type::<Foo>()
        .with_handoff(target("test_1"))
        .with_handoff(target("test_2"));
    let result = run(agent).await.unwrap();
    assert_eq!(
        result.final_output_as::<Foo>().unwrap(),
        Foo { bar: "456".into() }
    );
}

#[tokio::test]
async fn execute_handoffs_uses_public_agent_for_ignored_extra_handoffs() {
    let agent = Agent::new("source")
        .with_model_instance(scripted(vec![vec![
            handoff_call("test_1"),
            handoff_call("test_2"),
        ]]))
        .with_handoff(target("test_1"))
        .with_handoff(target("test_2"));
    let result = run(agent).await.unwrap();
    let ignored = result
        .new_items
        .iter()
        .find(|i| {
            matches!(
                i.kind,
                RunItemKind::HandoffOutput { .. } | RunItemKind::ToolCallOutput { .. }
            ) && raw_item_of(i)["call_id"] == "h-test_2"
        })
        .expect("ignored handoff output");
    assert_eq!(ignored.agent, "source");
    assert!(
        ignored.to_input_item().0["output"]
            .as_str()
            .unwrap()
            .contains("Multiple handoffs detected, ignoring this one.")
    );
}

#[tokio::test]
async fn execute_handoffs_preserves_tool_input_guardrail_results() {
    let guardrail = ToolInputGuardrail::new("check", |_data| async move {
        Ok(ToolGuardrailFunctionOutput::allow().with_output_info(json!("checked")))
    });
    let agent = Agent::new("triage")
        .with_model_instance(scripted(vec![vec![
            call("guarded", &json!({"value": "hi"}).to_string(), "c1"),
            handoff_call("target"),
        ]]))
        .with_tool(echo_tool("guarded").with_input_guardrail(guardrail))
        .with_handoff(target("target"));
    let result = run(agent).await.unwrap();
    assert_eq!(result.last_agent.name, "target");
    assert_eq!(
        result.tool_input_guardrail_results[0].output.output_info,
        json!("checked")
    );
}

fn raw_item_of(item: &RunItem) -> Value {
    item.to_input_item().0
}

// ---------------------------------------------------------------------------------------------
// Tracing of function tool errors and approvals
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn function_tool_error_trace_respects_sensitive_data_setting() {
    let log = Log::default();
    let agent = Agent::new("test")
        .with_model_instance(scripted(vec![vec![call("error_tool_trace", "{}", "1")]]))
        .with_tool(
            logging_tool("error_tool_trace", &log, Err("secret-token-123"))
                .without_failure_error_function(),
        );
    let err = run_with(
        agent,
        "hello",
        RunConfig::default().with_trace_include_sensitive_data(false),
    )
    .await
    .unwrap_err();
    assert_user_error(err, "Error running tool error_tool_trace: secret-token-123");
    let spans = function_spans("error_tool_trace");
    assert_eq!(spans.len(), 1);
    let error = &spans[0]["error"];
    assert_eq!(error["message"], "Error running tool");
    assert_eq!(error["data"]["tool_name"], "error_tool_trace");
    assert_eq!(
        error["data"]["error"],
        "Tool execution failed. Error details are redacted."
    );
    assert!(!error.to_string().contains("secret-token-123"));
}

#[tokio::test]
async fn default_function_tool_error_trace_respects_sensitive_data_setting() {
    let log = Log::default();
    let agent = Agent::new("test")
        .with_model_instance(scripted(vec![
            vec![call("error_tool_default", "{}", "1")],
            vec![assistant_message("done")],
        ]))
        .with_tool(logging_tool(
            "error_tool_default",
            &log,
            Err("secret-token-123"),
        ));
    let result = run_with(
        agent,
        "hello",
        RunConfig::default().with_trace_include_sensitive_data(false),
    )
    .await
    .unwrap();
    let items = first_turn(&result);
    assert_eq!(items.len(), 2);
    assert_function_output(
        &items[1],
        "An error occurred while running the tool. Please try again.",
    );
    let spans = function_spans("error_tool_default");
    assert_eq!(spans.len(), 1);
    let error = &spans[0]["error"];
    assert_eq!(error["message"], "Error running tool (non-fatal)");
    assert_eq!(error["data"]["tool_name"], "error_tool_default");
    assert_eq!(
        error["data"]["error"],
        "Tool execution failed. Error details are redacted."
    );
    assert!(!error.to_string().contains("secret-token-123"));
}

fn approval_tool(name: &str) -> FunctionTool<()> {
    FunctionTool::new_no_args(name, "", |_c| async move { Ok("ok") }).with_needs_approval(true)
}

#[tokio::test]
async fn pending_approval_function_span_output_excludes_internal_result_object() {
    let agent = Agent::new("test")
        .with_instructions("system-prompt-abc")
        .with_model_instance(scripted(vec![vec![call(
            "approval_tool_pending",
            "{}",
            "1",
        )]]))
        .with_tool(approval_tool("approval_tool_pending"));
    let result = run_with(
        agent,
        "hello",
        RunConfig::default().with_trace_include_sensitive_data(true),
    )
    .await
    .unwrap();
    assert_eq!(result.interruptions.len(), 1);
    for span in function_spans("approval_tool_pending") {
        assert!(span["span_data"]["output"].is_null());
        assert!(!span.to_string().contains("system-prompt-abc"));
    }
}

async fn rejected_span(tool_name: &str, include: bool, message: &str) -> Value {
    let model = scripted(vec![
        vec![call(tool_name, "{}", "1")],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::new("test")
        .with_model_instance(model)
        .with_tool(approval_tool(tool_name));
    let first = run(agent.clone()).await.unwrap();
    let mut state = first.to_state();
    let item = state.get_interruptions()[0].clone();
    state.reject(&item, false, Some(message.into()));
    run_with(
        agent,
        state,
        RunConfig::default().with_trace_include_sensitive_data(include),
    )
    .await
    .unwrap();
    let spans = function_spans(tool_name);
    assert_eq!(spans.len(), 1, "{spans:?}");
    spans[0].clone()
}

#[tokio::test]
async fn rejected_tool_function_span_output_respects_sensitive_data_setting() {
    let span = rejected_span("approval_tool_redacted", false, "secret-denial-456").await;
    assert!(span["span_data"]["output"].is_null());
    assert_eq!(span["error"]["message"], "Tool execution rejected");
    assert_eq!(span["error"]["data"]["tool_name"], "approval_tool_redacted");
    assert_eq!(
        span["error"]["data"]["error"],
        "Tool execution for 1 was manually rejected by user."
    );
    assert!(!span.to_string().contains("secret-denial-456"));
}

#[tokio::test]
async fn rejected_tool_function_span_keeps_rejection_message_when_sensitive_data_included() {
    let span = rejected_span("approval_tool_sensitive", true, "denied-by-policy").await;
    assert_eq!(span["span_data"]["output"], "denied-by-policy");
    assert_eq!(span["error"]["message"], "denied-by-policy");
}

#[tokio::test]
async fn execute_function_tool_calls_collapse_trace_name_for_top_level_deferred_tools() {
    let tool = FunctionTool::new_no_args("get_shipping_eta_a", "", |_c| async move { Ok("eta") })
        .with_defer_loading(true);
    let agent = Agent::new("test")
        .with_model_instance(scripted(vec![
            vec![function_call_full(
                "get_shipping_eta_a",
                r#"{"tracking_number":"ZX-123"}"#,
                "call-1",
                None,
                Some("get_shipping_eta_a".into()),
            )],
            vec![assistant_message("done")],
        ]))
        .with_tool(tool);
    run(agent).await.unwrap();
    assert_eq!(function_spans("get_shipping_eta_a").len(), 1);
    assert!(function_spans("get_shipping_eta_a.get_shipping_eta_a").is_empty());
}

#[tokio::test]
async fn execute_function_tool_calls_preserve_trace_name_for_explicit_namespace() {
    let tool = FunctionTool::new_no_args("get_shipping_eta_b", "", |_c| async move { Ok("eta") })
        .with_defer_loading(true);
    let tools =
        openai_agents_sdk::tool::tool_namespace("shipping", "Shipping tools", vec![tool]).unwrap();
    let agent = Agent::new("test")
        .with_model_instance(scripted(vec![
            vec![function_call_full(
                "get_shipping_eta_b",
                r#"{"tracking_number":"ZX-123"}"#,
                "call-1",
                None,
                Some("shipping".into()),
            )],
            vec![assistant_message("done")],
        ]))
        .with_tools(tools);
    run(agent).await.unwrap();
    assert_eq!(function_spans("shipping.get_shipping_eta_b").len(), 1);
    assert!(function_spans("get_shipping_eta_b").is_empty());
}

#[tokio::test]
async fn execute_function_tool_calls_rejects_reserved_same_name_namespace_shape() {
    let tool = FunctionTool::<()>::new_no_args("lookup_account", "", |_c| async move { Ok("x") })
        .with_defer_loading(true);
    let err = openai_agents_sdk::tool::tool_namespace(
        "lookup_account",
        "Same-name namespace",
        vec![tool],
    )
    .unwrap_err();
    assert_user_error(err, "synthetic namespace `lookup_account.lookup_account`");
}

// ---------------------------------------------------------------------------------------------
// Approvals
// ---------------------------------------------------------------------------------------------

#[derive(Default, Clone)]
struct RecordingEditor;

#[async_trait]
impl ApplyPatchEditor for RecordingEditor {
    async fn create_file(
        &self,
        _op: &ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<ApplyPatchResult> {
        Ok(ApplyPatchResult::completed("created"))
    }
    async fn update_file(
        &self,
        _op: &ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<ApplyPatchResult> {
        Ok(ApplyPatchResult::completed("updated"))
    }
    async fn delete_file(
        &self,
        _op: &ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<ApplyPatchResult> {
        Ok(ApplyPatchResult::completed("deleted"))
    }
}

fn apply_patch_call(call_id: &str) -> Value {
    json!({"type": "apply_patch_call", "call_id": call_id, "status": "completed",
           "operation": {"type": "update_file", "path": "test.md", "diff": "-a\n+b\n"}})
}

fn local_shell_call(call_id: &str) -> Value {
    json!({"type": "shell_call", "id": "shell_call", "call_id": call_id, "status": "completed",
           "action": {"commands": ["echo hi"], "timeout_ms": 1000}})
}

async fn single_interruption(agent: Agent<()>, expected_tool_name: &str) {
    let result = run(agent).await.unwrap();
    assert_eq!(result.interruptions.len(), 1);
    assert_eq!(result.interruptions[0].tool_name, expected_tool_name);
    assert!(result.final_output.is_none());
}

#[tokio::test]
async fn execute_tools_handles_tool_approval_items_function_tool() {
    single_interruption(
        Agent::new("test")
            .with_model_instance(scripted(vec![vec![call("test_tool", "{}", "c1")]]))
            .with_tool(approval_tool("test_tool")),
        "test_tool",
    )
    .await;
}

#[tokio::test]
async fn execute_tools_handles_tool_approval_items_shell_tool() {
    single_interruption(
        Agent::new("test")
            .with_model_instance(scripted(vec![vec![OutputItem::from_value(
                local_shell_call("call_shell"),
            )]]))
            .with_tool(
                ShellTool::new(|_req| async move { Ok("output") }).with_needs_approval(true),
            ),
        "shell",
    )
    .await;
}

#[tokio::test]
async fn execute_tools_handles_tool_approval_items_apply_patch_tool() {
    single_interruption(
        Agent::new("test")
            .with_model_instance(scripted(vec![vec![OutputItem::from_value(
                apply_patch_call("call_apply"),
            )]]))
            .with_tool(ApplyPatchTool::new(RecordingEditor).with_needs_approval(true)),
        "apply_patch",
    )
    .await;
}

async fn empty_call_id_case(raw: Value, tool: openai_agents_sdk::Tool<()>, calls: Log) {
    let agent = Agent::new("test")
        .with_model_instance(scripted(vec![vec![OutputItem::from_value(raw)]]))
        .with_tool(tool);
    let err = run(agent).await.unwrap_err();
    assert!(
        matches!(err, AgentsError::ModelBehaviorError { .. }),
        "{err:?}"
    );
    assert!(
        calls.lock().unwrap().is_empty(),
        "approval callback must not run"
    );
}

#[tokio::test]
async fn empty_action_call_id_fails_before_approval_callback_shell() {
    let calls = Log::default();
    let calls_in = calls.clone();
    let tool = ShellTool::new(|_req| async move { Ok("output") })
        .with_needs_approval(true)
        .with_on_approval(move |_ctx, _item| {
            calls_in.lock().unwrap().push("shell".into());
            async move { Some(ApprovalDecision::approve()) }
        });
    let mut raw = local_shell_call("");
    raw["id"] = json!("item-shell");
    empty_call_id_case(raw, tool.into(), calls).await;
}

#[tokio::test]
async fn empty_action_call_id_fails_before_approval_callback_apply_patch() {
    let calls = Log::default();
    let calls_in = calls.clone();
    let tool = ApplyPatchTool::new(RecordingEditor)
        .with_needs_approval(true)
        .with_on_approval(move |_ctx, _item| {
            calls_in.lock().unwrap().push("apply_patch".into());
            async move { Some(ApprovalDecision::approve()) }
        });
    let mut raw = apply_patch_call("");
    raw["id"] = json!("item-apply");
    empty_call_id_case(raw, tool.into(), calls).await;
}

#[tokio::test]
async fn execute_tools_preserves_synthetic_namespace_for_deferred_top_level_approval() {
    let tool = approval_tool("get_weather").with_defer_loading(true);
    let result = run(Agent::new("test")
        .with_model_instance(scripted(vec![vec![function_call_full(
            "get_weather",
            "{}",
            "c1",
            None,
            Some("get_weather".into()),
        )]]))
        .with_tool(tool))
    .await
    .unwrap();
    assert_eq!(result.interruptions.len(), 1);
    let interruption = &result.interruptions[0];
    assert_eq!(interruption.tool_name, "get_weather");
    assert_eq!(interruption.raw_item.extra["namespace"], "get_weather");
}

#[tokio::test]
async fn deferred_tool_approval_allows_bare_alias_when_visible_peer_is_disabled() {
    let visible = approval_tool("get_weather").with_is_enabled(false);
    let deferred = FunctionTool::new_no_args("get_weather", "", |_c| async move { Ok("deferred") })
        .with_needs_approval(true)
        .with_defer_loading(true);
    let model = scripted(vec![
        vec![function_call_full(
            "get_weather",
            "{}",
            "c1",
            None,
            Some("get_weather".into()),
        )],
        vec![assistant_message("done")],
    ]);
    let agent = Agent::new("test")
        .with_model_instance(model)
        .with_tool(visible)
        .with_tool(deferred);
    let first = run(agent.clone()).await.unwrap();
    assert_eq!(first.interruptions.len(), 1);
    assert_eq!(first.interruptions[0].tool_name, "get_weather");
    assert_eq!(
        first.interruptions[0].raw_item.extra["namespace"],
        "get_weather"
    );
    // The bare-name alias lets an approval granted for `get_weather` cover the deferred call.
    let mut state = first.to_state();
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);
    let resumed = run_with(agent, state, RunConfig::default()).await.unwrap();
    assert!(
        resumed
            .new_items
            .iter()
            .any(|i| matches!(&i.kind, RunItemKind::ToolCallOutput { raw_item, .. } if raw_item["output"] == "deferred"))
    );
}

// ---------------------------------------------------------------------------------------------
// Hosted MCP approvals
// ---------------------------------------------------------------------------------------------

fn hosted_mcp(callback: Option<Arc<AtomicUsize>>) -> HostedMCPTool<()> {
    let tool = HostedMCPTool::new(json!({
        "type": "mcp", "server_label": "test_mcp_server", "server_url": "https://example.com",
        "require_approval": "always", "allowed_callers": ["direct", "programmatic"]
    }));
    match callback {
        None => tool,
        Some(count) => tool.with_on_approval_request(move |_req| {
            count.fetch_add(1, Ordering::SeqCst);
            async move {
                MCPToolApprovalResult {
                    approve: true,
                    reason: None,
                }
            }
        }),
    }
}

fn mcp_request(id: &str, caller: Option<Value>) -> Value {
    let mut v = json!({"id": id, "type": "mcp_approval_request", "server_label": "test_mcp_server",
                       "arguments": "{}", "name": "list_repo_languages"});
    if let Some(caller) = caller {
        v["caller"] = caller;
    }
    v
}

#[tokio::test]
async fn execute_tools_runs_hosted_mcp_callback_when_present() {
    let count = Arc::new(AtomicUsize::new(0));
    let program_caller = json!({"type": "program", "caller_id": "program-1"});
    let agent = Agent::new("test")
        .with_model_instance(scripted(vec![
            vec![
                OutputItem::from_value(json!({"type": "program", "id": "program-1",
                    "call_id": "program-1", "code": "x", "fingerprint": "f"})),
                OutputItem::from_value(mcp_request("mcp-approval-1", Some(program_caller.clone()))),
            ],
            vec![assistant_message("done")],
        ]))
        .with_tool(hosted_mcp(Some(count.clone())))
        .with_tool(openai_agents_sdk::tool::ProgrammaticToolCallingTool);
    let result = run(agent).await.unwrap();
    assert!(result.interruptions.is_empty());
    assert_eq!(count.load(Ordering::SeqCst), 1);
    let response = result
        .new_items
        .iter()
        .find_map(|i| match &i.kind {
            RunItemKind::MCPApprovalResponse { raw_item } => Some(raw_item.clone()),
            _ => None,
        })
        .expect("approval response");
    assert_eq!(response["caller"], program_caller);
}

#[tokio::test]
async fn execute_tools_surfaces_hosted_mcp_interruptions_without_callback() {
    let agent = Agent::new("test")
        .with_model_instance(scripted(vec![vec![OutputItem::from_value(mcp_request(
            "mcp-approval-2",
            None,
        ))]]))
        .with_tool(hosted_mcp(None));
    let result = run(agent).await.unwrap();
    assert_eq!(result.interruptions.len(), 1);
    // The Rust approval item carries the request id as its call id.
    assert_eq!(result.interruptions[0].call_id(), "mcp-approval-2");
}

#[tokio::test]
async fn execute_tools_uses_public_agent_for_hosted_mcp_callback_results() {
    let count = Arc::new(AtomicUsize::new(0));
    let agent = Agent::new("public")
        .with_model_instance(scripted(vec![
            vec![OutputItem::from_value(mcp_request(
                "mcp-approval-callback-public-agent",
                None,
            ))],
            vec![assistant_message("done")],
        ]))
        .with_tool(hosted_mcp(Some(count)));
    let result = run(agent).await.unwrap();
    assert!(result.new_items.iter().any(|i| {
        matches!(i.kind, RunItemKind::MCPApprovalResponse { .. }) && i.agent == "public"
    }));
}

async fn mcp_rejection(reason: Option<&str>) -> Value {
    let agent = Agent::new("test")
        .with_model_instance(scripted(vec![
            vec![OutputItem::from_value(mcp_request("mcp-reject", None))],
            vec![assistant_message("done")],
        ]))
        .with_tool(hosted_mcp(None));
    let first = run(agent.clone()).await.unwrap();
    let mut state = first.to_state();
    let item = state.get_interruptions()[0].clone();
    state.reject(&item, false, reason.map(str::to_string));
    let resumed = run_with(agent, state, RunConfig::default()).await.unwrap();
    resumed
        .new_items
        .iter()
        .find_map(|i| match &i.kind {
            RunItemKind::MCPApprovalResponse { raw_item } => Some(raw_item.clone()),
            _ => None,
        })
        .expect("rejection response")
}

#[tokio::test]
async fn execute_tools_emits_hosted_mcp_rejection_response() {
    let response = mcp_rejection(None).await;
    assert_eq!(response["approve"], false);
    assert_eq!(response["approval_request_id"], "mcp-reject");
    assert!(response.get("reason").is_none());
}

#[tokio::test]
async fn execute_tools_emits_hosted_mcp_rejection_reason_from_explicit_message() {
    let response = mcp_rejection(Some("Denied by policy")).await;
    assert_eq!(response["approve"], false);
    assert_eq!(response["reason"], "Denied by policy");
}
