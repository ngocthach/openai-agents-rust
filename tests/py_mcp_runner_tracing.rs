//! Parity port of `tests/mcp/test_runner_calls_mcp.py` and `tests/mcp/test_mcp_tracing.py`,
//! and the guardrail/approval-order test of `tests/mcp/test_mcp_approval.py`.
//!
//! Every runner test runs both the plain and the streamed path (Python's `streaming`
//! parametrization). Tracing processors are process-global, so the tracing tests share one
//! lock and one processor.
#![allow(clippy::field_reassign_with_default)] // mirrors Python fixtures that tweak one field of a default config
#![cfg(feature = "mcp")]

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::error::{AgentsError, Result};
use openai_agents_sdk::items::RunItemKind;
use openai_agents_sdk::mcp::{MCPServer, MCPServerConfig, MCPTool, MCPToolPolicy, RequireApproval};
use openai_agents_sdk::result::RunResult;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::{RunConfig, ToolExecutionConfig};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_full};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool_guardrails::{
    ToolGuardrailFunctionOutput, ToolInputGuardrail, ToolOutputGuardrail,
};
use openai_agents_sdk::tracing::{InMemoryTraceProcessor, TracingProcessor, set_trace_processors};
use openai_agents_sdk::{Agent, MCPConfig, Runner};
use serde_json::{Map, Value, json};

// ---------------------------------------------------------------------------
// Fake server (tests/mcp/helpers.py::FakeMCPServer)
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct Fake {
    name: String,
    config: MCPServerConfig,
    tools: Mutex<Vec<(String, Value)>>,
    calls: Mutex<Vec<String>>,
    results: Mutex<Vec<String>>,
    crash: bool,
}

impl Fake {
    fn new() -> Self {
        Fake::named("fake_mcp_server")
    }
    fn named(name: &str) -> Self {
        Fake {
            name: name.into(),
            config: MCPServerConfig::default(),
            tools: Mutex::new(vec![]),
            calls: Mutex::new(vec![]),
            results: Mutex::new(vec![]),
            crash: false,
        }
    }
    fn crashing() -> Self {
        Fake {
            crash: true,
            ..Fake::new()
        }
    }
    fn tool(self, name: &str, schema: Value) -> Self {
        self.tools.lock().unwrap().push((name.into(), schema));
        self
    }
    fn calls(&self) -> Vec<String> {
        self.calls.lock().unwrap().clone()
    }
}

#[async_trait::async_trait]
impl MCPServer for Fake {
    fn name(&self) -> &str {
        &self.name
    }
    fn config(&self) -> &MCPServerConfig {
        &self.config
    }
    async fn connect(&self) -> Result<()> {
        Ok(())
    }
    async fn cleanup(&self) -> Result<()> {
        Ok(())
    }
    async fn list_tools(&self) -> Result<Vec<MCPTool>> {
        Ok(self
            .tools
            .lock()
            .unwrap()
            .iter()
            .map(|(n, s)| MCPTool {
                meta: None,
                name: n.clone(),
                title: None,
                description: None,
                input_schema: s.clone(),
                annotations: None,
            })
            .collect())
    }
    async fn call_tool(&self, name: &str, args: Option<Map<String, Value>>) -> Result<Value> {
        self.calls.lock().unwrap().push(name.into());
        if self.crash {
            return Err(AgentsError::other("Crash!"));
        }
        let args = args
            .map(|a| Value::Object(a).to_string())
            .unwrap_or_else(|| "{}".into());
        let text = format!("result_{name}_{args}");
        self.results.lock().unwrap().push(text.clone());
        Ok(json!({"content": [{"type": "text", "text": text}]}))
    }
    async fn list_prompts(&self) -> Result<Vec<Value>> {
        Ok(vec![])
    }
    async fn get_prompt(&self, _n: &str, _a: Option<BTreeMap<String, String>>) -> Result<Value> {
        Ok(Value::Null)
    }
}

fn call(name: &str, args: &str, id: &str) -> openai_agents_sdk::items::OutputItem {
    function_call_full(name, args, id, None, None)
}

/// Run `agent` (streamed or not) and return the final `RunResult`.
async fn run(
    agent: &Agent<()>,
    input: &str,
    streaming: bool,
    options: RunOptions<()>,
) -> Result<RunResult<()>> {
    if streaming {
        let mut s = Runner::run_streamed_with_options(agent.clone(), input, options);
        {
            let events = s.stream_events();
            futures::pin_mut!(events);
            while let Some(event) = events.next().await {
                event?;
            }
        }
        Ok(s.into_result().expect("a finished stream has a result"))
    } else {
        Runner::run_with_options(agent.clone(), input, options).await
    }
}

fn agent_with(model: &Arc<FakeModel>, servers: Vec<Arc<dyn MCPServer>>) -> Agent<()> {
    let mut agent = Agent::<()>::new("test").with_model_instance(model.clone());
    for s in servers {
        agent = agent.with_mcp_server(s);
    }
    agent
}

fn tool_outputs(model: &FakeModel) -> Vec<Value> {
    let last = model.calls().pop().expect("a model call");
    last.input
        .iter()
        .filter(|i| i.as_value().get("type") == Some(&json!("function_call_output")))
        .map(|i| i.as_value()["output"].clone())
        .collect()
}

// ---------------------------------------------------------------------------
// test_runner_calls_mcp.py
// ---------------------------------------------------------------------------

#[tokio::test]
async fn runner_calls_mcp_tool() {
    let _guard = TRACE_LOCK.lock().await;
    for streaming in [false, true] {
        let server = Arc::new(
            Fake::new()
                .tool("test_tool_1", json!({}))
                .tool("test_tool_2", json!({}))
                .tool("test_tool_3", json!({})),
        );
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![
            assistant_message("a_message"),
            call("test_tool_2", "", "c1"),
        ]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = agent_with(&model, vec![server.clone()]);
        run(&agent, "user_message", streaming, RunOptions::default())
            .await
            .unwrap();
        assert_eq!(server.calls(), ["test_tool_2"], "streaming={streaming}");
    }
}

#[tokio::test]
async fn runner_errors_when_the_mcp_tool_is_not_found() {
    let _guard = TRACE_LOCK.lock().await;
    for streaming in [false, true] {
        let server = Arc::new(Fake::new().tool("test_tool_1", json!({})));
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![
            assistant_message("a_message"),
            call("test_tool_doesnt_exist", "", "c1"),
        ]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = agent_with(&model, vec![server]);
        let error = run(&agent, "user_message", streaming, RunOptions::default())
            .await
            .unwrap_err();
        assert!(
            matches!(error, AgentsError::ModelBehaviorError { .. }),
            "streaming={streaming}: {error}"
        );
    }
}

#[tokio::test]
async fn runner_works_with_multiple_mcp_servers() {
    let _guard = TRACE_LOCK.lock().await;
    for streaming in [false, true] {
        let s1 = Arc::new(Fake::new().tool("test_tool_1", json!({})));
        let s2 = Arc::new(
            Fake::new()
                .tool("test_tool_2", json!({}))
                .tool("test_tool_3", json!({})),
        );
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![
            assistant_message("a_message"),
            call("test_tool_2", "", "c1"),
        ]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = agent_with(&model, vec![s1.clone(), s2.clone()]);
        run(&agent, "user_message", streaming, RunOptions::default())
            .await
            .unwrap();
        assert!(s1.calls().is_empty());
        assert_eq!(s2.calls(), ["test_tool_2"]);
    }
}

#[tokio::test]
async fn runner_errors_when_mcp_tools_clash() {
    let _guard = TRACE_LOCK.lock().await;
    for streaming in [false, true] {
        let s1 = Arc::new(
            Fake::new()
                .tool("test_tool_1", json!({}))
                .tool("test_tool_2", json!({})),
        );
        let s2 = Arc::new(
            Fake::new()
                .tool("test_tool_2", json!({}))
                .tool("test_tool_3", json!({})),
        );
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![
            assistant_message("a_message"),
            call("test_tool_3", "", "c1"),
        ]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = agent_with(&model, vec![s1, s2]);
        let error = run(&agent, "user_message", streaming, RunOptions::default())
            .await
            .unwrap_err();
        assert!(
            matches!(error, AgentsError::UserError { .. }),
            "streaming={streaming}: {error}"
        );
    }
}

fn prefixed_config() -> MCPConfig<()> {
    MCPConfig::<()>::default().with_include_server_in_tool_names(true)
}

#[tokio::test]
async fn runner_can_call_server_prefixed_tool_names() {
    let _guard = TRACE_LOCK.lock().await;
    for streaming in [false, true] {
        let docs = Arc::new(Fake::named("docs").tool("search", json!({})));
        let calendar = Arc::new(Fake::named("calendar").tool("search", json!({})));
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![
            assistant_message("a_message"),
            call("mcp_calendar__search", "", "c1"),
        ]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = agent_with(&model, vec![docs.clone(), calendar.clone()])
            .with_mcp_config(prefixed_config());
        run(&agent, "user_message", streaming, RunOptions::default())
            .await
            .unwrap();
        assert!(docs.calls().is_empty());
        assert_eq!(calendar.calls(), ["search"]);
    }
}

#[tokio::test]
async fn prefixed_mcp_names_do_not_collide_with_agent_tools() {
    let _guard = TRACE_LOCK.lock().await;
    for streaming in [false, true] {
        let docs = Arc::new(Fake::named("docs").tool("search", json!({})));
        let calendar = Arc::new(Fake::named("calendar").tool("search", json!({})));
        let local_calls = Arc::new(Mutex::new(Vec::<String>::new()));
        let sink = local_calls.clone();
        let local = FunctionTool::<()>::new_raw(
            "mcp_calendar__search",
            "Local tool that intentionally collides with the natural MCP prefix.",
            json!({"type": "object", "properties": {}, "additionalProperties": false}),
            move |_ctx, args: String| {
                let sink = sink.clone();
                async move {
                    sink.lock().unwrap().push(args);
                    Ok(openai_agents_sdk::tool::ToolOutput::text("local"))
                }
            },
        );
        let model = Arc::new(FakeModel::new());
        let agent = agent_with(&model, vec![docs.clone(), calendar.clone()])
            .with_tool(local)
            .with_mcp_config(prefixed_config());
        let agent_arc = Arc::new(agent.clone());
        let tools = agent_arc
            .get_all_tools(&openai_agents_sdk::run_context::RunContext::new(()))
            .await
            .unwrap();
        let names: Vec<String> = tools.iter().map(|t| t.name()).collect();
        let calendar_name = names
            .iter()
            .find(|n| n.starts_with("mcp_calendar__search_"))
            .expect("the colliding MCP tool gets a hash-suffixed name")
            .clone();
        assert!(
            names.contains(&"mcp_calendar__search".to_string()),
            "the local tool keeps its name"
        );

        model.add_turn(vec![
            assistant_message("a_message"),
            call(&calendar_name, "", "c1"),
        ]);
        model.add_turn(vec![assistant_message("done")]);
        run(&agent, "user_message", streaming, RunOptions::default())
            .await
            .unwrap();
        assert!(local_calls.lock().unwrap().is_empty());
        assert!(docs.calls().is_empty());
        assert_eq!(calendar.calls(), ["search"]);
    }
}

#[tokio::test]
async fn prefixed_mcp_names_do_not_collide_with_handoffs() {
    let _guard = TRACE_LOCK.lock().await;
    for streaming in [false, true] {
        let server = Arc::new(Fake::named("calendar").tool("search", json!({})));
        let target_model = Arc::new(FakeModel::new());
        let target = Agent::<()>::new("calendar_agent").with_model_instance(target_model.clone());
        let handoff =
            openai_agents_sdk::handoffs::handoff(target).with_tool_name("mcp_calendar__search");
        let model = Arc::new(FakeModel::new());
        let agent = agent_with(&model, vec![server.clone()])
            .with_handoff(handoff)
            .with_mcp_config(prefixed_config());

        let tools = Arc::new(agent.clone())
            .get_all_tools(&openai_agents_sdk::run_context::RunContext::new(()))
            .await
            .unwrap();
        let mcp_names: Vec<String> = tools
            .iter()
            .map(|t| t.name())
            .filter(|n| n.starts_with("mcp_calendar__search"))
            .collect();
        assert_eq!(mcp_names.len(), 1);
        assert_ne!(mcp_names[0], "mcp_calendar__search");
        assert!(mcp_names[0].starts_with("mcp_calendar__search_"));

        model.add_turn(vec![
            assistant_message("a_message"),
            call(&mcp_names[0], "", "c1"),
        ]);
        model.add_turn(vec![assistant_message("done")]);
        run(&agent, "user_message", streaming, RunOptions::default())
            .await
            .unwrap();
        assert_eq!(server.calls(), ["search"]);
        assert!(
            target_model.calls().is_empty(),
            "the handoff target must not run"
        );
    }
}

#[tokio::test]
async fn runner_calls_mcp_tool_with_args() {
    let _guard = TRACE_LOCK.lock().await;
    for streaming in [false, true] {
        let schema = json!({
            "properties": {"bar": {"type": "string"}, "baz": {"type": "integer"}},
            "required": ["bar", "baz"], "type": "object",
        });
        let server = Arc::new(
            Fake::new()
                .tool("test_tool_1", json!({}))
                .tool("test_tool_2", schema)
                .tool("test_tool_3", json!({})),
        );
        let json_args = r#"{"bar":"baz","baz":1}"#;
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![
            assistant_message("a_message"),
            call("test_tool_2", json_args, "c1"),
        ]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = agent_with(&model, vec![server.clone()]);
        run(&agent, "user_message", streaming, RunOptions::default())
            .await
            .unwrap();
        assert_eq!(server.calls(), ["test_tool_2"]);
        assert_eq!(
            *server.results.lock().unwrap(),
            [format!("result_test_tool_2_{json_args}")]
        );
    }
}

#[tokio::test]
async fn runner_emits_an_error_output_item_when_the_mcp_tool_raises() {
    let _guard = TRACE_LOCK.lock().await;
    for streaming in [false, true] {
        let server = Arc::new(Fake::crashing().tool("crashing_tool", json!({})));
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![
            assistant_message("a_message"),
            call("crashing_tool", "{}", "c1"),
        ]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = agent_with(&model, vec![server]);
        let result = run(&agent, "user_message", streaming, RunOptions::default())
            .await
            .unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("done"));
        let outputs: Vec<&Value> = result
            .new_items
            .iter()
            .filter_map(|i| match &i.kind {
                RunItemKind::ToolCallOutput { output, .. } => Some(output),
                _ => None,
            })
            .collect();
        assert!(
            !outputs.is_empty(),
            "expected a tool_call_output_item for the MCP failure"
        );
        assert_eq!(
            outputs[0],
            &json!("An error occurred while running the tool. Please try again.")
        );
    }
}

fn guarded_server(
    input: Option<ToolInputGuardrail<()>>,
    output: Option<ToolOutputGuardrail<()>>,
    tool: &str,
) -> Arc<Fake> {
    let policy = MCPToolPolicy::<()> {
        tool_input_guardrails: input.into_iter().collect(),
        tool_output_guardrails: output.into_iter().collect(),
        ..Default::default()
    };
    let mut fake = Fake::new().tool(tool, json!({}));
    fake.config = fake.config.with_tool_policy(policy);
    Arc::new(fake)
}

#[tokio::test]
async fn an_input_guardrail_rejection_prevents_the_server_call() {
    let _guard = TRACE_LOCK.lock().await;
    for streaming in [false, true] {
        let seen = Arc::new(Mutex::new(Vec::<(String, String)>::new()));
        let s = seen.clone();
        let guardrail = ToolInputGuardrail::<()>::new("reject_input", move |data| {
            let s = s.clone();
            async move {
                s.lock().unwrap().push((
                    data.context.tool_name.clone(),
                    data.context.tool_arguments.clone(),
                ));
                Ok(ToolGuardrailFunctionOutput::reject_content(
                    "blocked MCP input",
                ))
            }
        });
        let server = guarded_server(Some(guardrail), None, "sensitive");
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![call("sensitive", r#"{"secret":"value"}"#, "c1")]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = agent_with(&model, vec![server.clone()]);
        let result = run(&agent, "user_message", streaming, RunOptions::default())
            .await
            .unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("done"));
        assert!(server.calls().is_empty());
        assert_eq!(
            *seen.lock().unwrap(),
            [("sensitive".to_string(), r#"{"secret":"value"}"#.to_string())]
        );
        assert_eq!(result.tool_input_guardrail_results.len(), 1);
        assert_eq!(tool_outputs(&model), [json!("blocked MCP input")]);
    }
}

#[tokio::test]
async fn an_output_guardrail_checks_the_converted_output_before_the_model_sees_it() {
    let _guard = TRACE_LOCK.lock().await;
    for streaming in [false, true] {
        let seen = Arc::new(Mutex::new(Vec::<String>::new()));
        let s = seen.clone();
        let guardrail = ToolOutputGuardrail::<()>::new("reject_output", move |data| {
            let s = s.clone();
            async move {
                s.lock().unwrap().push(data.output.to_model_string());
                Ok(ToolGuardrailFunctionOutput::reject_content(
                    "blocked MCP output",
                ))
            }
        });
        let server = guarded_server(None, Some(guardrail), "lookup");
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![call("lookup", "{}", "c1")]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = agent_with(&model, vec![server.clone()]);
        let result = run(&agent, "user_message", streaming, RunOptions::default())
            .await
            .unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("done"));
        assert_eq!(server.calls(), ["lookup"]);
        let tool_result = server.results.lock().unwrap()[0].clone();
        assert_eq!(*seen.lock().unwrap(), std::slice::from_ref(&tool_result));
        assert_eq!(result.tool_output_guardrail_results.len(), 1);
        assert_eq!(tool_outputs(&model), [json!("blocked MCP output")]);
        assert!(!format!("{:?}", model.calls().last().unwrap().input).contains(&tool_result));
    }
}

// ---------------------------------------------------------------------------
// test_mcp_approval.py: guardrails with approval and serialized resume
// ---------------------------------------------------------------------------

async fn guardrails_with_approval(pre_approval: bool) {
    let _guard = TRACE_LOCK.lock().await;
    for streaming in [false, true] {
        let seen = Arc::new(Mutex::new(Vec::<(String, String)>::new()));
        let s = seen.clone();
        let guardrail = ToolInputGuardrail::<()>::new("allow_input", move |data| {
            let s = s.clone();
            async move {
                s.lock().unwrap().push((
                    data.context.tool_name.clone(),
                    data.context.tool_arguments.clone(),
                ));
                Ok(ToolGuardrailFunctionOutput::allow())
            }
        });
        let policy = MCPToolPolicy::<()> {
            tool_input_guardrails: vec![guardrail],
            ..Default::default()
        };
        let mut fake = Fake::new().tool("add", json!({"type": "object", "properties": {}}));
        fake.config = fake.config.with_tool_policy(policy);
        fake.config.require_approval = RequireApproval::Always;
        let server = Arc::new(fake);
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![call("add", "{}", "guarded_mcp_call")]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = agent_with(&model, vec![server.clone()]);
        let options = || {
            RunOptions::default().run_config(RunConfig::default().with_tool_execution(
                ToolExecutionConfig {
                    pre_approval_tool_input_guardrails: pre_approval,
                    ..Default::default()
                },
            ))
        };

        let first = run(&agent, "call add", streaming, options()).await.unwrap();
        assert_eq!(first.interruptions.len(), 1);
        assert!(server.calls().is_empty());
        let expected: Vec<(String, String)> = if pre_approval {
            vec![("add".into(), "{}".into())]
        } else {
            vec![]
        };
        assert_eq!(
            *seen.lock().unwrap(),
            expected,
            "streaming={streaming} pre={pre_approval}"
        );

        let mut state = first.to_state();
        let item = state.get_interruptions()[0].clone();
        state.approve(&item, false);
        let restored =
            openai_agents_sdk::run_state::RunState::from_json_str(&state.to_json_string().unwrap())
                .unwrap();
        let resumed = if streaming {
            let mut s = Runner::run_streamed_with_options(agent.clone(), restored, options());
            {
                let ev = s.stream_events();
                futures::pin_mut!(ev);
                while let Some(e) = ev.next().await {
                    e.unwrap();
                }
            }
            s.into_result().unwrap()
        } else {
            Runner::run_with_options(agent.clone(), restored, options())
                .await
                .unwrap()
        };
        assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
        assert_eq!(server.calls(), ["add"]);
        let runs = if pre_approval { 2 } else { 1 };
        assert_eq!(
            seen.lock().unwrap().len(),
            runs,
            "streaming={streaming} pre_approval={pre_approval}"
        );
    }
}

#[tokio::test]
async fn guardrails_preserve_approval_and_serialized_resume_order() {
    guardrails_with_approval(false).await;
}

#[tokio::test]
async fn pre_approval_guardrails_run_once_before_and_once_after_approval() {
    guardrails_with_approval(true).await;
}

// ---------------------------------------------------------------------------
// test_mcp_tracing.py
// ---------------------------------------------------------------------------

/// Tracing processors are global, so the tracing tests run one at a time on one processor.
static TRACE_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn function_span_for(items: &[Value], name: &str) -> Value {
    items
        .iter()
        .find(|i| {
            i["object"] == "trace.span"
                && i["span_data"]["type"] == "function"
                && i["span_data"]["name"] == name
        })
        .unwrap_or_else(|| panic!("no function span named {name}"))
        .clone()
}

fn spans_of(items: &[Value], kind: &str) -> Vec<Value> {
    items
        .iter()
        .filter(|i| i["object"] == "trace.span" && i["span_data"]["type"] == kind)
        .cloned()
        .collect()
}

#[tokio::test]
async fn mcp_tracing_records_tool_listing_and_mcp_data() {
    let _guard = TRACE_LOCK.lock().await;
    let processor = Arc::new(InMemoryTraceProcessor::new());
    set_trace_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);

    let server = Arc::new(Fake::new().tool("test_tool_1", json!({})));
    let model = Arc::new(FakeModel::new());
    let non_mcp = FunctionTool::<()>::new_raw(
        "non_mcp_tool",
        "",
        json!({"type": "object", "properties": {}}),
        |_c, _a: String| async { Ok(openai_agents_sdk::tool::ToolOutput::text("tool_result")) },
    );
    let agent = agent_with(&model, vec![server.clone()]).with_tool(non_mcp);

    model.add_turn(vec![
        assistant_message("a_message"),
        call("test_tool_1", "", "mcp_call_1"),
    ]);
    model.add_turn(vec![assistant_message("done")]);
    let result = run(&agent, "first_test", true, RunOptions::default())
        .await
        .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));

    let items = processor.items();
    let listings = spans_of(&items, "mcp_tools");
    assert!(
        !listings.is_empty(),
        "tool discovery records an mcp_tools span"
    );
    assert!(
        listings
            .iter()
            .all(|s| s["span_data"]["server"] == "fake_mcp_server"
                && s["span_data"]["result"] == json!(["test_tool_1"]))
    );
    let agent_span = &spans_of(&items, "agent")[0];
    assert_eq!(
        agent_span["span_data"]["tools"],
        json!(["test_tool_1", "non_mcp_tool"])
    );

    let f = function_span_for(&items, "test_tool_1");
    assert_eq!(
        f["span_data"]["mcp_data"],
        json!({"server": "fake_mcp_server"})
    );
    assert_eq!(f["span_data"]["output"], "result_test_tool_1_{}");

    // A server that gains tools shows them in the next listing; non-MCP tool spans carry no
    // mcp_data.
    processor.clear();
    server
        .tools
        .lock()
        .unwrap()
        .push(("test_tool_2".into(), json!({})));
    model.add_turn(vec![
        assistant_message("a_message"),
        call("non_mcp_tool", "", "function_call_1"),
        call("test_tool_2", "", "mcp_call_2"),
    ]);
    model.add_turn(vec![assistant_message("done")]);
    run(&agent, "second_test", false, RunOptions::default())
        .await
        .unwrap();
    let items = processor.items();
    assert_eq!(
        spans_of(&items, "agent")[0]["span_data"]["tools"],
        json!(["test_tool_1", "test_tool_2", "non_mcp_tool"])
    );
    assert!(
        spans_of(&items, "mcp_tools")
            .iter()
            .all(|s| s["span_data"]["result"] == json!(["test_tool_1", "test_tool_2"]))
    );
    let plain = function_span_for(&items, "non_mcp_tool");
    assert_eq!(plain["span_data"]["output"], "tool_result");
    assert!(
        plain["span_data"]
            .get("mcp_data")
            .is_none_or(Value::is_null)
    );
    assert_eq!(
        function_span_for(&items, "test_tool_2")["span_data"]["mcp_data"],
        json!({"server": "fake_mcp_server"})
    );
}

#[tokio::test]
async fn mcp_tracing_redacts_output_when_sensitive_data_is_disabled() {
    let _guard = TRACE_LOCK.lock().await;
    let processor = Arc::new(InMemoryTraceProcessor::new());
    set_trace_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);
    let server = Arc::new(Fake::new().tool("test_tool_1", json!({})));
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![call("test_tool_1", "", "c1")]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = agent_with(&model, vec![server]);
    let mut config = RunConfig::<()>::default();
    config.trace_include_sensitive_data = false;
    run(
        &agent,
        "redaction_test",
        false,
        RunOptions::default().run_config(config),
    )
    .await
    .unwrap();

    let items = processor.items();
    let f = function_span_for(&items, "test_tool_1");
    assert!(
        f["span_data"].get("output").is_none_or(Value::is_null),
        "output is redacted: {f}"
    );
    assert!(
        f["span_data"].get("input").is_none_or(Value::is_null),
        "input is redacted: {f}"
    );
    assert_eq!(
        f["span_data"]["mcp_data"],
        json!({"server": "fake_mcp_server"})
    );
}

#[tokio::test]
async fn mcp_tracing_always_hides_url_credentials() {
    for include_sensitive in [true, false] {
        let _guard = TRACE_LOCK.lock().await;
        let processor = Arc::new(InMemoryTraceProcessor::new());
        set_trace_processors(vec![processor.clone() as Arc<dyn TracingProcessor>]);
        let server = Arc::new(
            Fake::named(
                "streamable_http: https://user:s3cr3t_pw@mcp.example.test:8443/mcp?api_key=SECRET_QS_KEY#SECRET_FRAGMENT",
            )
            .tool("search", json!({})),
        );
        let model = Arc::new(FakeModel::new());
        model.add_turn(vec![call("search", "", "c1")]);
        model.add_turn(vec![assistant_message("done")]);
        let agent = agent_with(&model, vec![server]);
        let mut config = RunConfig::<()>::default();
        config.trace_include_sensitive_data = include_sensitive;
        run(
            &agent,
            "trace_url_credentials",
            false,
            RunOptions::default().run_config(config),
        )
        .await
        .unwrap();

        let serialized = serde_json::to_string(&processor.items()).unwrap();
        let safe = "streamable_http: https://mcp.example.test:8443/mcp";
        assert_eq!(
            serialized.matches(safe).count(),
            3,
            "include_sensitive={include_sensitive}: {serialized}"
        );
        for secret in ["user", "s3cr3t_pw", "SECRET_QS_KEY", "SECRET_FRAGMENT"] {
            assert!(!serialized.contains(secret), "{secret} leaked");
        }
    }
}
