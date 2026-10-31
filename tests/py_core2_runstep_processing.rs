//! Ported from the Python SDK's `tests/test_run_step_processing.py` and
//! `tests/test_process_model_response.py`.
//!
//! Python calls `run_loop.process_model_response` / `execute_handoffs` directly. Rust's
//! response classification is private, so each port feeds the same model output through
//! `Runner` and asserts what the classification decides: the run items it records, which tool
//! runs (and with which arguments), which handoff happens, and the error it raises.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use openai_agents_sdk::computer::{Computer, Environment};
use openai_agents_sdk::editor::{ApplyPatchEditor, ApplyPatchOperation, ApplyPatchResult};
use openai_agents_sdk::handoffs::{default_tool_description, default_tool_name};
use openai_agents_sdk::items::{OutputItem, RunItemKind};
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full, function_call_with_id,
};
use openai_agents_sdk::tool::{
    ApplyPatchTool, ComputerTool, CustomTool, FunctionTool, ShellTool, tool_namespace,
};
use openai_agents_sdk::{
    Agent, AgentsError, RunConfig, RunItem, RunOptions, RunResult, Runner, ToolNotFoundBehavior,
    handoff,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

type Log = Arc<Mutex<Vec<String>>>;

fn disable_tracing() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
}

fn scripted(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    let model = Arc::new(ScriptedModel::new());
    for step in steps {
        model.enqueue(ScriptedStep::output(step)).unwrap();
    }
    model
}

/// A function tool that records `name:arguments` and returns `result`.
fn recording_tool(name: &str, result: &str, log: &Log) -> FunctionTool<()> {
    let log = log.clone();
    let tool_name = name.to_string();
    let result = result.to_string();
    FunctionTool::new_raw(
        name,
        "",
        json!({"type": "object", "properties": {}}),
        move |_c, args: String| {
            log.lock().unwrap().push(format!("{tool_name}:{args}"));
            let result = result.clone();
            async move { Ok(result) }
        },
    )
}

/// A `lookup`-style tool returning `prefix:<customer_id>`.
fn prefixed_tool(name: &str, prefix: &str) -> FunctionTool<()> {
    let prefix = prefix.to_string();
    FunctionTool::new(name, "", move |_c, args: Value| {
        let id = args
            .get("customer_id")
            .or_else(|| args.get("city"))
            .or_else(|| args.get("payload"))
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string();
        let out = if prefix.is_empty() {
            id
        } else {
            format!("{prefix}:{id}")
        };
        async move { Ok(out) }
    })
}

fn call(name: &str, args: &str, call_id: &str, namespace: Option<&str>) -> OutputItem {
    function_call_full(name, args, call_id, None, namespace.map(str::to_string))
}

async fn run(
    agent: Agent<()>,
    config: Option<RunConfig<()>>,
) -> openai_agents_sdk::Result<RunResult<()>> {
    disable_tracing();
    let mut options = RunOptions::new(());
    if let Some(config) = config {
        options = options.run_config(config);
    }
    Runner::run_with_options(&agent, "hello", options).await
}

fn tool_calls(result: &RunResult<()>) -> Vec<&RunItem> {
    result
        .new_items
        .iter()
        .filter(|i| matches!(i.kind, RunItemKind::ToolCall { .. }))
        .collect()
}

fn tool_call_raw(result: &RunResult<()>, index: usize) -> Value {
    match &tool_calls(result)[index].kind {
        RunItemKind::ToolCall { raw_item } => raw_item.clone(),
        _ => unreachable!(),
    }
}

fn tool_outputs(result: &RunResult<()>) -> Vec<Value> {
    result
        .new_items
        .iter()
        .filter_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { raw_item, .. } => Some(raw_item.clone()),
            _ => None,
        })
        .collect()
}

fn function_outputs(result: &RunResult<()>) -> Vec<Value> {
    tool_outputs(result)
        .into_iter()
        .filter(|o| o["type"] == "function_call_output")
        .map(|o| o["output"].clone())
        .collect()
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
// test_run_step_processing.py
// ---------------------------------------------------------------------------------------------

#[tokio::test]
async fn empty_response() {
    let agent = Agent::<()>::new("test").with_model_instance(scripted(vec![vec![]]));
    let result = run(agent, None).await.unwrap();
    assert!(result.new_items.is_empty());
    assert_eq!(result.final_output_text().as_deref(), Some(""));
}

#[tokio::test]
async fn no_tool_calls() {
    let agent = Agent::<()>::new("test")
        .with_model_instance(scripted(vec![vec![assistant_message("Hello, world!")]]));
    let result = run(agent, None).await.unwrap();
    assert!(tool_calls(&result).is_empty());
    assert_eq!(result.final_output_text().as_deref(), Some("Hello, world!"));
}

#[tokio::test]
async fn single_tool_call() {
    let log = Log::default();
    let agent = Agent::<()>::new("test")
        .with_model_instance(scripted(vec![
            vec![
                assistant_message("Hello, world!"),
                call("test", "", "c1", None),
            ],
            vec![assistant_message("done")],
        ]))
        .with_tool(recording_tool("test", "ok", &log));
    let result = run(agent, None).await.unwrap();
    assert_eq!(*log.lock().unwrap(), vec!["test:"]);
    assert_eq!(tool_call_raw(&result, 0)["name"], "test");
    assert_eq!(tool_call_raw(&result, 0)["arguments"], "");
}

#[tokio::test]
async fn missing_tool_call_raises_error() {
    let log = Log::default();
    let agent = Agent::<()>::new("test")
        .with_model_instance(scripted(vec![vec![
            assistant_message("Hello, world!"),
            call("missing", "", "c1", None),
        ]]))
        .with_tool(recording_tool("test", "ok", &log));
    assert_model_behavior(run(agent, None).await.unwrap_err(), "missing");
}

#[tokio::test]
async fn multiple_tool_calls() {
    let log = Log::default();
    let agent = Agent::<()>::new("test")
        .with_model_instance(scripted(vec![
            vec![
                assistant_message("Hello, world!"),
                call("test_1", "abc", "c1", None),
                call("test_2", "xyz", "c2", None),
            ],
            vec![assistant_message("done")],
        ]))
        .with_tool(recording_tool("test_1", "1", &log))
        .with_tool(recording_tool("test_2", "2", &log))
        .with_tool(recording_tool("test_3", "3", &log));
    let result = run(agent, None).await.unwrap();
    let mut seen = log.lock().unwrap().clone();
    seen.sort();
    assert_eq!(seen, vec!["test_1:abc", "test_2:xyz"]);
    assert_eq!(tool_call_raw(&result, 0)["name"], "test_1");
    assert_eq!(tool_call_raw(&result, 0)["arguments"], "abc");
    assert_eq!(tool_call_raw(&result, 1)["name"], "test_2");
    assert_eq!(tool_call_raw(&result, 1)["arguments"], "xyz");
}

fn handoff_call(target: &str, call_id: &str) -> OutputItem {
    function_call_with_id(default_tool_name(target), "{}", call_id)
}

#[tokio::test]
async fn handoffs_parsed_correctly() {
    disable_tracing();
    let a1 = Arc::new(
        Agent::<()>::new("test_1")
            .with_model_instance(scripted(vec![vec![assistant_message("from test_1")]])),
    );
    let a2 = Arc::new(Agent::<()>::new("test_2").with_model_instance(scripted(vec![])));
    let model = scripted(vec![
        vec![assistant_message("Hello, world!")],
        vec![
            assistant_message("Hello, world!"),
            handoff_call("test_1", "h1"),
        ],
    ]);
    let a3 = Agent::<()>::new("test_3")
        .with_model_instance(model.clone())
        .with_handoff(a1.clone())
        .with_handoff(a2.clone());
    let first = Runner::run(&a3, "hi").await.unwrap();
    assert_eq!(first.last_agent.name, "test_3", "no handoff for plain text");

    let second = Runner::run(&a3, "hi").await.unwrap();
    assert_eq!(second.last_agent.name, "test_1");
    let request = model.last_call().unwrap().request;
    let spec = request
        .handoffs
        .iter()
        .find(|h| h.name == default_tool_name("test_1"))
        .expect("handoff spec");
    assert_eq!(spec.description, default_tool_description("test_1", None));
    assert!(
        second
            .new_items
            .iter()
            .any(|i| matches!(i.kind, RunItemKind::HandoffCall { .. }))
    );
}

async fn nesting_case(handoff_nest: bool, run_nest: bool) -> Vec<Value> {
    disable_tracing();
    let target_model = scripted(vec![vec![assistant_message("target done")]]);
    let target = Arc::new(Agent::<()>::new("target").with_model_instance(target_model.clone()));
    let source = Agent::<()>::new("source")
        .with_model_instance(scripted(vec![vec![handoff_call("target", "h1")]]))
        .with_handoff(handoff(target).with_nest_handoff_history(handoff_nest));
    Runner::run_with_options(
        &source,
        "hello",
        RunOptions::new(()).run_config(RunConfig::default().with_nest_handoff_history(run_nest)),
    )
    .await
    .unwrap();
    target_model
        .first_call()
        .unwrap()
        .request
        .input
        .into_iter()
        .map(|i| i.0)
        .collect()
}

#[tokio::test]
async fn handoff_can_disable_run_level_history_nesting() {
    let input = nesting_case(false, true).await;
    // Not nested: the target sees the original input verbatim, then the handoff pair.
    assert_eq!(input[0], json!({"role": "user", "content": "hello"}));
    assert!(input.iter().any(|i| i["type"] == "function_call"));
}

#[tokio::test]
async fn handoff_can_enable_history_nesting() {
    let input = nesting_case(true, false).await;
    // Nested: the history collapses into one summary message.
    assert!(
        input
            .iter()
            .all(|i| i.get("type") != Some(&json!("function_call")))
    );
    assert!(
        input
            .iter()
            .any(|i| i["role"] == "assistant" || i["role"] == "user"),
        "{input:?}"
    );
    assert_ne!(input[0], json!({"role": "user", "content": "hello"}));
}

#[tokio::test]
async fn missing_handoff_fails() {
    let a1 = Arc::new(Agent::<()>::new("test_1").with_model_instance(scripted(vec![])));
    let a3 = Agent::<()>::new("test_3")
        .with_model_instance(scripted(vec![vec![
            assistant_message("Hello, world!"),
            handoff_call("test_2", "h1"),
        ]]))
        .with_handoff(a1);
    assert_model_behavior(run(a3, None).await.unwrap_err(), "transfer_to_test_2");
}

#[tokio::test]
async fn multiple_handoffs_doesnt_error() {
    let a1 = Arc::new(
        Agent::<()>::new("test_1")
            .with_model_instance(scripted(vec![vec![assistant_message("one")]])),
    );
    let a2 = Arc::new(Agent::<()>::new("test_2").with_model_instance(scripted(vec![])));
    let a3 = Agent::<()>::new("test_3")
        .with_model_instance(scripted(vec![vec![
            assistant_message("Hello, world!"),
            handoff_call("test_1", "h1"),
            handoff_call("test_2", "h2"),
        ]]))
        .with_handoff(a1)
        .with_handoff(a2);
    let result = run(a3, None).await.unwrap();
    let handoff_calls = result
        .new_items
        .iter()
        .filter(|i| matches!(i.kind, RunItemKind::HandoffCall { .. }))
        .count();
    assert_eq!(handoff_calls, 2);
    assert_eq!(result.last_agent.name, "test_1");
}

#[derive(Debug, Serialize, Deserialize, JsonSchema, PartialEq)]
struct Foo {
    bar: String,
}

#[tokio::test]
async fn final_output_parsed_correctly() {
    let agent = Agent::<()>::new("test")
        .with_model_instance(scripted(vec![vec![
            assistant_message("Hello, world!"),
            assistant_message(r#"{"bar":"123"}"#),
        ]]))
        .with_output_type::<Foo>();
    let result = run(agent, None).await.unwrap();
    assert_eq!(
        result.final_output_as::<Foo>().unwrap(),
        Foo { bar: "123".into() }
    );
}

async fn hosted_call_case(raw: Value) {
    let agent = Agent::<()>::new("test").with_model_instance(scripted(vec![vec![
        assistant_message("hello"),
        OutputItem::from_value(raw.clone()),
    ]]));
    let result = run(agent, None).await.unwrap();
    assert_eq!(tool_calls(&result).len(), 1);
    assert_eq!(tool_call_raw(&result, 0), raw);
    assert!(tool_outputs(&result).is_empty(), "no tool run is scheduled");
    assert_eq!(result.final_output_text().as_deref(), Some("hello"));
}

#[tokio::test]
async fn file_search_tool_call_parsed_correctly() {
    hosted_call_case(
        json!({"id": "fs1", "queries": ["query"], "status": "completed",
                            "type": "file_search_call"}),
    )
    .await;
}

#[tokio::test]
async fn function_web_search_tool_call_parsed_correctly() {
    hosted_call_case(
        json!({"id": "w1", "action": {"type": "search", "query": "query"},
                            "status": "completed", "type": "web_search_call"}),
    )
    .await;
}

#[tokio::test]
async fn reasoning_item_parsed_correctly() {
    let reasoning = json!({"id": "r1", "type": "reasoning",
                           "summary": [{"text": "why", "type": "summary_text"}]});
    let agent = Agent::<()>::new("test")
        .with_model_instance(scripted(vec![vec![OutputItem::from_value(reasoning)]]));
    let result = run(agent, None).await.unwrap();
    let item = result
        .new_items
        .iter()
        .find_map(|i| match &i.kind {
            RunItemKind::Reasoning { raw_item } => Some(raw_item.clone()),
            _ => None,
        })
        .expect("reasoning item");
    assert_eq!(item.id, "r1");
    assert_eq!(
        item.summary,
        vec![json!({"text": "why", "type": "summary_text"})]
    );
}

#[derive(Default)]
struct DummyComputer {
    log: Log,
}

#[async_trait]
impl Computer for DummyComputer {
    fn environment(&self) -> Option<Environment> {
        Some(Environment::Mac)
    }
    fn dimensions(&self) -> Option<(u32, u32)> {
        Some((0, 0))
    }
    async fn screenshot(&self) -> openai_agents_sdk::Result<String> {
        Ok(String::new())
    }
    async fn click(
        &self,
        x: i64,
        y: i64,
        b: &str,
        _k: Option<&[String]>,
    ) -> openai_agents_sdk::Result<()> {
        self.log.lock().unwrap().push(format!("click {x} {y} {b}"));
        Ok(())
    }
    async fn double_click(
        &self,
        _: i64,
        _: i64,
        _: Option<&[String]>,
    ) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn scroll(
        &self,
        _: i64,
        _: i64,
        _: i64,
        _: i64,
        _: Option<&[String]>,
    ) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn type_text(&self, _: &str) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn wait(&self) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn move_mouse(
        &self,
        _: i64,
        _: i64,
        _: Option<&[String]>,
    ) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn keypress(&self, _: &[String]) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
    async fn drag(&self, _: &[(i64, i64)], _: Option<&[String]>) -> openai_agents_sdk::Result<()> {
        Ok(())
    }
}

fn computer_call() -> Value {
    json!({"id": "c1", "type": "computer_call", "call_id": "c1",
           "action": {"type": "click", "x": 1, "y": 2, "button": "left"},
           "pending_safety_checks": [], "status": "completed"})
}

#[tokio::test]
async fn computer_tool_call_without_computer_tool_raises_error() {
    let agent =
        Agent::<()>::new("test").with_model_instance(scripted(vec![vec![OutputItem::from_value(
            computer_call(),
        )]]));
    assert_model_behavior(run(agent, None).await.unwrap_err(), "computer");
}

#[tokio::test]
async fn computer_tool_call_with_computer_tool_parsed_correctly() {
    let computer = DummyComputer::default();
    let log = computer.log.clone();
    let agent = Agent::<()>::new("test")
        .with_model_instance(scripted(vec![
            vec![OutputItem::from_value(computer_call())],
            vec![assistant_message("done")],
        ]))
        .with_tool(ComputerTool::new(computer));
    let result = run(agent, None).await.unwrap();
    assert_eq!(tool_call_raw(&result, 0), computer_call());
    assert_eq!(*log.lock().unwrap(), vec!["click 1 2 left"]);
}

#[tokio::test]
async fn tool_and_handoff_parsed_correctly() {
    let log = Log::default();
    let a1 = Arc::new(
        Agent::<()>::new("test_1")
            .with_model_instance(scripted(vec![vec![assistant_message("one")]])),
    );
    let a2 = Arc::new(Agent::<()>::new("test_2").with_model_instance(scripted(vec![])));
    let a3 = Agent::<()>::new("test_3")
        .with_model_instance(scripted(vec![vec![
            assistant_message("Hello, world!"),
            call("test", "abc", "c1", None),
            handoff_call("test_1", "h1"),
        ]]))
        .with_tool(recording_tool("test", "ok", &log))
        .with_handoff(a1)
        .with_handoff(a2);
    let result = run(a3, None).await.unwrap();
    assert_eq!(*log.lock().unwrap(), vec!["test:abc"]);
    assert_eq!(result.last_agent.name, "test_1");
}

// ---------------------------------------------------------------------------------------------
// test_process_model_response.py
// ---------------------------------------------------------------------------------------------

fn shell_call(call_id: &str) -> Value {
    json!({"type": "shell_call", "id": format!("sh_{call_id}"), "call_id": call_id,
           "status": "completed", "action": {"commands": ["echo hi"], "timeout_ms": 1000}})
}

fn hosted_shell() -> ShellTool<()> {
    ShellTool::hosted(json!({"type": "container_auto"})).unwrap()
}

#[tokio::test]
async fn process_model_response_shell_call_without_tool_raises() {
    let agent = Agent::<()>::new("no-shell").with_model_instance(scripted(vec![vec![
        OutputItem::from_value(shell_call("shell-1")),
    ]]));
    assert_model_behavior(run(agent, None).await.unwrap_err(), "shell tool");
}

#[tokio::test]
async fn process_model_response_skips_local_shell_execution_for_hosted_environment() {
    let agent = Agent::<()>::new("hosted-shell")
        .with_model_instance(scripted(vec![
            vec![OutputItem::from_value(shell_call("shell-hosted-1"))],
            vec![assistant_message("done")],
        ]))
        .with_tool(hosted_shell());
    let result = run(agent, None).await.unwrap();
    assert_eq!(tool_calls(&result).len(), 1);
    assert!(
        tool_outputs(&result).is_empty(),
        "hosted shells never run locally"
    );
}

#[tokio::test]
async fn process_model_response_sanitizes_shell_call_model_object() {
    let mut raw = shell_call("call_shell_2");
    raw["created_by"] = json!("server");
    let agent = Agent::<()>::new("hosted-shell-model")
        .with_model_instance(scripted(vec![
            vec![OutputItem::from_value(raw)],
            vec![assistant_message("done")],
        ]))
        .with_tool(hosted_shell());
    let result = run(agent, None).await.unwrap();
    let item = tool_call_raw(&result, 0);
    assert_eq!(item["type"], "shell_call");
    assert!(item.get("created_by").is_none());
    let next = tool_calls(&result)[0].to_input_item().0;
    assert_eq!(next["type"], "shell_call");
    assert!(next.get("created_by").is_none());
    assert!(tool_outputs(&result).is_empty());
}

fn shell_output(created_by: bool) -> Value {
    let mut chunk =
        json!({"stdout": "ok\n", "stderr": "", "outcome": {"type": "exit", "exit_code": 0}});
    let mut out = json!({"type": "shell_call_output", "id": "sh_out_1", "call_id": "call_shell_1",
                         "status": "completed", "max_output_length": 1000});
    if created_by {
        chunk["created_by"] = json!("server");
        out["created_by"] = json!("server");
    }
    out["output"] = json!([chunk]);
    out
}

#[tokio::test]
async fn process_model_response_preserves_shell_call_output() {
    let agent = Agent::<()>::new("shell-output").with_model_instance(scripted(vec![
        vec![OutputItem::from_value(shell_output(false))],
        vec![assistant_message("done")],
    ]));
    let result = run(agent, None).await.unwrap();
    assert_eq!(tool_outputs(&result), vec![shell_output(false)]);
}

#[tokio::test]
async fn process_model_response_sanitizes_shell_call_output_model_object() {
    let agent = Agent::<()>::new("shell-output-model").with_model_instance(scripted(vec![
        vec![OutputItem::from_value(shell_output(true))],
        vec![assistant_message("done")],
    ]));
    let result = run(agent, None).await.unwrap();
    let raw = &tool_outputs(&result)[0];
    assert_eq!(raw["type"], "shell_call_output");
    assert!(raw.get("created_by").is_none());
    assert!(raw["output"][0].get("created_by").is_none());
    let item = result
        .new_items
        .iter()
        .find(|i| matches!(i.kind, RunItemKind::ToolCallOutput { .. }))
        .unwrap();
    let next = item.to_input_item().0;
    assert_eq!(next["type"], "shell_call_output");
    assert!(next.get("status").is_none());
    assert!(next.get("created_by").is_none());
    assert!(next["output"][0].get("created_by").is_none());
}

#[derive(Default, Clone)]
struct RecordingEditor {
    ops: Log,
}

#[async_trait]
impl ApplyPatchEditor for RecordingEditor {
    async fn create_file(
        &self,
        op: &ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<ApplyPatchResult> {
        self.ops.lock().unwrap().push(format!("create {}", op.path));
        Ok(ApplyPatchResult::completed("created"))
    }
    async fn update_file(
        &self,
        op: &ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<ApplyPatchResult> {
        self.ops.lock().unwrap().push(format!("update {}", op.path));
        Ok(ApplyPatchResult::completed("updated"))
    }
    async fn delete_file(
        &self,
        op: &ApplyPatchOperation,
    ) -> openai_agents_sdk::Result<ApplyPatchResult> {
        self.ops.lock().unwrap().push(format!("delete {}", op.path));
        Ok(ApplyPatchResult::completed("deleted"))
    }
}

fn apply_patch_call(call_id: &str, created_by: bool) -> Value {
    let mut v = json!({"type": "apply_patch_call", "id": format!("ap_{call_id}"), "call_id": call_id,
                       "status": "completed",
                       "operation": {"type": "update_file", "path": "test.md", "diff": "-old\n+new\n"}});
    if created_by {
        v["created_by"] = json!("server");
    }
    v
}

#[tokio::test]
async fn process_model_response_apply_patch_call_without_tool_raises() {
    let agent = Agent::<()>::new("no-apply").with_model_instance(scripted(vec![vec![
        OutputItem::from_value(apply_patch_call("apply-1", false)),
    ]]));
    assert_model_behavior(run(agent, None).await.unwrap_err(), "apply_patch tool");
}

async fn apply_patch_case(call: Value) -> (RunResult<()>, Vec<String>) {
    let editor = RecordingEditor::default();
    let ops = editor.ops.clone();
    let agent = Agent::<()>::new("apply-agent")
        .with_model_instance(scripted(vec![
            vec![OutputItem::from_value(call)],
            vec![assistant_message("done")],
        ]))
        .with_tool(ApplyPatchTool::new(editor));
    let result = run(agent, None).await.unwrap();
    let ops = ops.lock().unwrap().clone();
    (result, ops)
}

#[tokio::test]
async fn process_model_response_sanitizes_apply_patch_call_model_object() {
    let (result, ops) = apply_patch_case(apply_patch_call("call_apply_1", true)).await;
    let raw = tool_call_raw(&result, 0);
    assert_eq!(raw["type"], "apply_patch_call");
    assert!(raw.get("created_by").is_none());
    let next = tool_calls(&result)[0].to_input_item().0;
    assert!(next.get("created_by").is_none());
    assert_eq!(ops, vec!["update test.md"], "the queued call ran");
}

#[tokio::test]
async fn process_model_response_queues_apply_patch_call() {
    let (result, ops) = apply_patch_case(apply_patch_call("apply-1", false)).await;
    assert_eq!(tool_call_raw(&result, 0)["type"], "apply_patch_call");
    assert_eq!(ops, vec!["update test.md"]);
}

#[tokio::test]
async fn process_model_response_queues_hosted_apply_patch_from_custom_tool_call() {
    let custom = json!({"type": "custom_tool_call", "name": "apply_patch", "call_id": "custom-apply-1",
                        "input": "{\"type\":\"update_file\",\"path\":\"test.md\",\"diff\":\"-old\\n+new\\n\"}"});
    let (result, ops) = apply_patch_case(custom).await;
    assert_eq!(tool_calls(&result).len(), 1);
    let raw = tool_call_raw(&result, 0);
    assert_eq!(raw["type"], "apply_patch_call");
    assert_eq!(raw["operation"]["type"], "update_file");
    assert_eq!(ops, vec!["update test.md"]);
}

#[tokio::test]
async fn process_model_response_queues_custom_tool_call_for_custom_tool() {
    let seen = Log::default();
    let seen_in = seen.clone();
    let custom_tool =
        CustomTool::<()>::new("raw_editor", "Edit raw text.", move |_c, input: String| {
            seen_in.lock().unwrap().push(input.clone());
            async move { Ok(input) }
        })
        .with_format(json!({"type": "text"}));
    let custom = json!({"type": "custom_tool_call", "name": "raw_editor", "call_id": "custom-apply-1",
                        "input": "-old\n+new\n"});
    let agent = Agent::<()>::new("custom-agent")
        .with_model_instance(scripted(vec![
            vec![OutputItem::from_value(custom.clone())],
            vec![assistant_message("done")],
        ]))
        .with_tool(custom_tool);
    let result = run(agent, None).await.unwrap();
    assert_eq!(tool_call_raw(&result, 0), custom);
    assert_eq!(*seen.lock().unwrap(), vec!["-old\n+new\n"]);
}

#[tokio::test]
async fn process_model_response_prefers_namespaced_function_over_apply_patch_fallback() {
    let tools = tool_namespace(
        "billing",
        "Billing tools",
        vec![prefixed_tool("apply_patch_lookup", "")],
    )
    .unwrap();
    let agent = Agent::<()>::new("billing-agent")
        .with_model_instance(scripted(vec![
            vec![call(
                "apply_patch_lookup",
                r#"{"payload":"value"}"#,
                "c1",
                Some("billing"),
            )],
            vec![assistant_message("done")],
        ]))
        .with_tools(tools);
    let result = run(agent, None).await.unwrap();
    assert_eq!(function_outputs(&result), vec![json!("value")]);
}

#[tokio::test]
async fn process_model_response_handles_compaction_item() {
    let agent = Agent::<()>::new("compaction-agent").with_model_instance(scripted(vec![vec![
        OutputItem::from_value(json!({"id": "comp-1", "encrypted_content": "enc",
                                      "type": "compaction", "created_by": "server"})),
    ]]));
    let result = run(agent, None).await.unwrap();
    let raw = result
        .new_items
        .iter()
        .find_map(|i| match &i.kind {
            RunItemKind::Compaction { raw_item } => Some(raw_item.clone()),
            _ => None,
        })
        .expect("compaction item");
    assert_eq!(raw["type"], "compaction");
    assert_eq!(raw["encrypted_content"], "enc");
    assert!(raw.get("created_by").is_none());
}

#[tokio::test]
async fn process_model_response_classifies_tool_search_items() {
    disable_tracing();
    let agent = Agent::<()>::new("tool-search-agent").with_model_instance(scripted(vec![
        vec![
            OutputItem::from_value(json!({"id": "tsc_123", "type": "tool_search_call",
                "arguments": {"paths": ["crm"], "query": "profile"}, "execution": "server",
                "status": "completed"})),
            OutputItem::from_value(json!({"id": "tso_123", "type": "tool_search_output",
                "execution": "server", "status": "completed",
                "tools": [{"type": "function", "name": "get_customer_profile",
                           "description": "Fetch a CRM customer profile.",
                           "parameters": {"type": "object",
                                          "properties": {"customer_id": {"type": "string"}},
                                          "required": ["customer_id"]},
                           "defer_loading": true}]})),
        ],
        vec![assistant_message("done")],
    ]));
    let mut streamed = Runner::run_streamed(&agent, "hello");
    let mut names = Vec::new();
    {
        use futures::StreamExt;
        let mut events = std::pin::pin!(streamed.stream_events());
        while let Some(ev) = events.next().await {
            if let StreamEvent::RunItem { name, .. } = ev.unwrap() {
                names.push(name);
            }
        }
    }
    assert_eq!(names[0], RunItemStreamEventName::ToolSearchCalled);
    assert_eq!(names[1], RunItemStreamEventName::ToolSearchOutputCreated);
}

fn crm_and_billing() -> Vec<FunctionTool<()>> {
    let mut tools = tool_namespace(
        "crm",
        "CRM tools",
        vec![prefixed_tool("lookup_account", "crm")],
    )
    .unwrap();
    tools.extend(
        tool_namespace(
            "billing",
            "Billing tools",
            vec![prefixed_tool("lookup_account", "billing")],
        )
        .unwrap(),
    );
    tools
}

#[tokio::test]
async fn process_model_response_uses_namespace_for_duplicate_function_names() {
    let agent = Agent::<()>::new("billing-agent")
        .with_model_instance(scripted(vec![
            vec![call(
                "lookup_account",
                r#"{"customer_id":"customer_42"}"#,
                "c1",
                Some("billing"),
            )],
            vec![assistant_message("done")],
        ]))
        .with_tools(crm_and_billing());
    let result = run(agent, None).await.unwrap();
    assert_eq!(
        function_outputs(&result),
        vec![json!("billing:customer_42")]
    );
}

fn deferred(name: &str, prefix: &str) -> FunctionTool<()> {
    prefixed_tool(name, prefix).with_defer_loading(true)
}

#[tokio::test]
async fn process_model_response_collapses_synthetic_deferred_namespace_in_tools_used() {
    let agent = Agent::<()>::new("weather-agent")
        .with_model_instance(scripted(vec![
            vec![call(
                "get_weather",
                r#"{"city":"Tokyo"}"#,
                "c1",
                Some("get_weather"),
            )],
            vec![assistant_message("done")],
        ]))
        .with_tool(deferred("get_weather", ""));
    let result = run(agent, None).await.unwrap();
    assert_eq!(function_outputs(&result), vec![json!("Tokyo")]);
}

#[tokio::test]
async fn process_model_response_rejects_bare_name_for_duplicate_namespaced_functions() {
    let agent = Agent::<()>::new("billing-agent")
        .with_model_instance(scripted(vec![vec![call(
            "lookup_account",
            r#"{"customer_id":"customer_42"}"#,
            "c1",
            None,
        )]]))
        .with_tools(crm_and_billing());
    assert_model_behavior(
        run(agent, None).await.unwrap_err(),
        "Tool lookup_account not found",
    );
}

#[tokio::test]
async fn process_model_response_uses_last_duplicate_top_level_function() {
    let agent = Agent::<()>::new("lookup-agent")
        .with_model_instance(scripted(vec![
            vec![call(
                "lookup",
                r#"{"customer_id":"customer_42"}"#,
                "c1",
                None,
            )],
            vec![assistant_message("done")],
        ]))
        .with_tool(prefixed_tool("lookup", "first"))
        .with_tool(prefixed_tool("lookup", "second"));
    let result = run(agent, None).await.unwrap();
    assert_eq!(function_outputs(&result), vec![json!("second:customer_42")]);
}

#[tokio::test]
async fn process_model_response_rejects_reserved_same_name_namespace_shape() {
    // Python builds this shape by assigning the private namespace attribute; the Rust
    // constructor is where the same reserved shape is refused.
    let err = tool_namespace(
        "lookup_account",
        "Same-name namespace",
        vec![prefixed_tool("lookup_account", "")],
    )
    .map(|tools| {
        Agent::<()>::new("lookup-agent")
            .with_model_instance(scripted(vec![vec![call(
                "lookup_account",
                r#"{"customer_id":"customer_42"}"#,
                "c1",
                Some("lookup_account"),
            )]]))
            .with_tools(tools)
    });
    let err = match err {
        Err(e) => e,
        Ok(agent) => run(agent, None).await.unwrap_err(),
    };
    match err {
        AgentsError::UserError { message, .. } => assert!(
            message.contains("synthetic namespace `lookup_account.lookup_account`"),
            "{message}"
        ),
        other => panic!("expected UserError, got {other:?}"),
    }
}

#[tokio::test]
async fn process_model_response_rejects_qualified_name_collision_with_dotted_top_level_tool() {
    let mut tools = vec![prefixed_tool("crm.lookup_account", "dotted")];
    tools.extend(
        tool_namespace(
            "crm",
            "CRM tools",
            vec![prefixed_tool("lookup_account", "crm")],
        )
        .unwrap(),
    );
    let agent = Agent::<()>::new("lookup-agent")
        .with_model_instance(scripted(vec![vec![call(
            "lookup_account",
            r#"{"customer_id":"customer_42"}"#,
            "c1",
            Some("crm"),
        )]]))
        .with_tools(tools);
    match run(agent, None).await.unwrap_err() {
        AgentsError::UserError { message, .. } => {
            assert!(
                message.contains("qualified name `crm.lookup_account`"),
                "{message}"
            )
        }
        other => panic!("expected UserError, got {other:?}"),
    }
}

#[tokio::test]
async fn process_model_response_prefers_visible_top_level_function_over_deferred_same_name_tool() {
    let agent = Agent::<()>::new("lookup-agent")
        .with_model_instance(scripted(vec![
            vec![call(
                "lookup_account",
                r#"{"customer_id":"customer_42"}"#,
                "c1",
                None,
            )],
            vec![assistant_message("done")],
        ]))
        .with_tool(prefixed_tool("lookup_account", "visible"))
        .with_tool(deferred("lookup_account", "deferred"));
    let result = run(agent, None).await.unwrap();
    assert_eq!(
        function_outputs(&result),
        vec![json!("visible:customer_42")]
    );
    assert!(tool_call_raw(&result, 0).get("namespace").is_none());
}

#[tokio::test]
async fn process_model_response_uses_internal_lookup_key_for_deferred_top_level_calls() {
    let agent = Agent::<()>::new("lookup-agent")
        .with_model_instance(scripted(vec![
            vec![call(
                "lookup_account",
                r#"{"customer_id":"customer_42"}"#,
                "c1",
                Some("lookup_account"),
            )],
            vec![assistant_message("done")],
        ]))
        .with_tool(prefixed_tool("lookup_account.lookup_account", "visible"))
        .with_tool(deferred("lookup_account", "deferred"));
    let result = run(agent, None).await.unwrap();
    assert_eq!(
        function_outputs(&result),
        vec![json!("deferred:customer_42")]
    );
}

#[tokio::test]
async fn process_model_response_preserves_synthetic_namespace_for_deferred_top_level_tools() {
    let agent = Agent::<()>::new("weather-agent")
        .with_model_instance(scripted(vec![
            vec![call(
                "get_weather",
                r#"{"city":"Tokyo"}"#,
                "c1",
                Some("get_weather"),
            )],
            vec![assistant_message("done")],
        ]))
        .with_tool(deferred("get_weather", ""));
    let result = run(agent, None).await.unwrap();
    assert_eq!(function_outputs(&result), vec![json!("Tokyo")]);
    assert_eq!(tool_call_raw(&result, 0)["namespace"], "get_weather");
}

#[tokio::test]
async fn process_model_response_prefers_namespaced_function_over_handoff_name_collision() {
    // The target owns a scripted model too, so a wrong handoff can never reach a real model.
    let target = Arc::new(
        Agent::<()>::new("lookup-agent")
            .with_model_instance(scripted(vec![vec![assistant_message("wrong")]])),
    );
    let agent = Agent::<()>::new("billing-agent")
        .with_model_instance(scripted(vec![
            vec![call(
                "lookup_account",
                r#"{"customer_id":"customer_42"}"#,
                "c1",
                Some("billing"),
            )],
            vec![assistant_message("done")],
        ]))
        .with_tools(
            tool_namespace(
                "billing",
                "Billing tools",
                vec![prefixed_tool("lookup_account", "billing")],
            )
            .unwrap(),
        )
        .with_handoff(handoff(target).with_tool_name("lookup_account"));
    let result = run(agent, None).await.unwrap();
    assert_eq!(
        function_outputs(&result),
        vec![json!("billing:customer_42")]
    );
    assert_eq!(result.last_agent.name, "billing-agent");
    assert!(
        result
            .new_items
            .iter()
            .all(|i| !matches!(i.kind, RunItemKind::HandoffCall { .. }))
    );
}

#[tokio::test]
async fn process_model_response_rejects_mismatched_function_namespace() {
    let agent = Agent::<()>::new("bare-agent")
        .with_model_instance(scripted(vec![vec![call(
            "lookup_account",
            r#"{"customer_id":"customer_42"}"#,
            "c1",
            Some("crm"),
        )]]))
        .with_tool(prefixed_tool("lookup_account", ""));
    assert_model_behavior(run(agent, None).await.unwrap_err(), "crm.lookup_account");
}

#[tokio::test]
async fn process_model_response_collects_missing_function_tool_when_opted_in() {
    let agent = Agent::<()>::new("test")
        .with_model_instance(scripted(vec![
            vec![call("missing_tool", "{}", "call_missing", None)],
            vec![assistant_message("done")],
        ]))
        .with_tool(prefixed_tool("ok_tool", ""));
    let config =
        RunConfig::default().with_tool_not_found_behavior(ToolNotFoundBehavior::ReturnErrorToModel);
    let result = run(agent, Some(config)).await.unwrap();
    assert_eq!(tool_calls(&result).len(), 1);
    assert_eq!(tool_call_raw(&result, 0)["name"], "missing_tool");
    let outputs = tool_outputs(&result);
    assert_eq!(outputs.len(), 1);
    assert_eq!(outputs[0]["call_id"], "call_missing");
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
}

#[tokio::test]
async fn process_model_response_uses_mcp_list_tools_metadata_for_hosted_mcp_calls() {
    let hosted = openai_agents_sdk::HostedMCPTool::<()>::new(json!({
        "type": "mcp", "server_label": "docs_server", "server_url": "https://example.com/mcp"
    }));
    let agent = Agent::<()>::new("hosted-mcp")
        .with_model_instance(scripted(vec![
            vec![
                OutputItem::from_value(json!({
                "id": "list_docs_server", "server_label": "docs_server", "type": "mcp_list_tools",
                "tools": [{"name": "search_docs", "input_schema": {}, "description": "Search the docs.",
                           "annotations": {"title": "Search Docs"}}]})),
                OutputItem::from_value(json!({"id": "mcp_call_1", "arguments": "{}", "name": "search_docs",
                                              "server_label": "docs_server", "type": "mcp_call",
                                              "status": "completed"})),
                assistant_message("done"),
            ],
        ]))
        .with_tool(hosted);
    let result = run(agent, None).await.unwrap();
    let mcp_call = result
        .new_items
        .iter()
        .find(|i| matches!(&i.kind, RunItemKind::ToolCall { raw_item } if raw_item["type"] == "mcp_call"))
        .expect("mcp call item");
    assert_eq!(mcp_call.description.as_deref(), Some("Search the docs."));
    assert_eq!(mcp_call.title.as_deref(), Some("Search Docs"));
}
