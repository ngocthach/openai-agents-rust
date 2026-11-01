#![cfg(feature = "mcp")]
//! Ports of `test_tool_exception_formatter.py`, `test_default_tool_error_redaction.py` and
//! `test_model_error_logging_redaction.py`.
//!
//! The logging policy flags, the `log` backend and the trace processors are process-global, so
//! every test holds one lock, resets the flags it touches and reads only the records produced
//! while it holds the lock.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, OnceLock};

use futures::StreamExt;
use openai_agents_sdk::debug::{
    reset_dont_log_model_data, reset_dont_log_tool_data, set_dont_log_model_data,
    set_dont_log_tool_data,
};
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::mcp::{MCPServer, MCPServerConfig, MCPTool, MCPToolPolicy};
use openai_agents_sdk::result::RunResult;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::{RunConfig, ToolErrorKind};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full,
};
use openai_agents_sdk::tool::{FunctionTool, default_tool_error_function, tool_namespace};
use openai_agents_sdk::tracing::{InMemoryTraceProcessor, TracingProcessor, set_trace_processors};
use openai_agents_sdk::{Agent, AgentAsToolOptions, AgentsError, MCPConfig, Result, Runner};
use serde_json::{Map, Value, json};

const SECRET: &str = "synthetic-private-exception-detail";
const SENTINEL: &str = "synthetic-private-tool-detail";
const GENERIC: &str = "An error occurred while running the tool. Please try again.";
const APPROVED: &str = "Order IDs must start with ORD-.";

// ---------------------------------------------------------------------------
// Shared fixtures: log capture, trace capture, flag guard
// ---------------------------------------------------------------------------

static RECORDS: Mutex<Vec<String>> = Mutex::new(Vec::new());
static LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        RECORDS.lock().unwrap().push(format!(
            "{} {}: {}",
            record.level(),
            record.target(),
            record.args()
        ));
    }
    fn flush(&self) {}
}

/// `(kind, name, call id, message, redacted text, raw text)` for each formatter call.
type Received = Arc<Mutex<Vec<(ToolErrorKind, String, String, String, String, String)>>>;

fn processor() -> &'static Arc<InMemoryTraceProcessor> {
    static PROCESSOR: OnceLock<Arc<InMemoryTraceProcessor>> = OnceLock::new();
    PROCESSOR.get_or_init(|| {
        let _ = log::set_logger(&Capture);
        log::set_max_level(log::LevelFilter::Trace);
        let p = Arc::new(InMemoryTraceProcessor::new());
        set_trace_processors(vec![p.clone() as Arc<dyn TracingProcessor>]);
        p
    })
}

/// Holds the global lock; restores the env-default logging policy when dropped.
struct Fixture {
    _guard: tokio::sync::MutexGuard<'static, ()>,
}

impl Drop for Fixture {
    fn drop(&mut self) {
        reset_dont_log_model_data();
        reset_dont_log_tool_data();
    }
}

async fn fixture() -> Fixture {
    let guard = LOCK.lock().await;
    openai_agents_sdk::tracing::set_tracing_disabled(false);
    processor().clear();
    RECORDS.lock().unwrap().clear();
    // Python's defaults: both data classes are redacted from logs.
    set_dont_log_model_data(true);
    set_dont_log_tool_data(true);
    Fixture { _guard: guard }
}

fn logs() -> String {
    RECORDS.lock().unwrap().join("\n")
}

fn spans() -> String {
    serde_json::to_string(&processor().items()).unwrap()
}

fn call(name: &str, args: &str, id: &str) -> OutputItem {
    function_call_full(name, args, id, None, None)
}

fn model(steps: Vec<Vec<OutputItem>>) -> Arc<ScriptedModel> {
    Arc::new(
        ScriptedModel::with_steps(steps.into_iter().map(ScriptedStep::output).collect()).unwrap(),
    )
}

/// `make_model`: one call of `name` with call id `call-5324`, then `done`.
fn make_model(name: &str) -> Arc<ScriptedModel> {
    model(vec![
        vec![call(name, "{}", "call-5324")],
        vec![assistant_message("done")],
    ])
}

fn last_input(model: &ScriptedModel) -> Vec<Value> {
    model
        .last_call()
        .expect("a model call")
        .request
        .input
        .iter()
        .map(|i| i.as_value().clone())
        .collect()
}

/// `model_output`: the first `function_call_output` of the last model call.
fn model_output(model: &ScriptedModel) -> Value {
    last_input(model)
        .into_iter()
        .find(|i| i["type"] == "function_call_output")
        .expect("a function_call_output")["output"]
        .clone()
}

async fn run(
    agent: Agent<()>,
    input: &str,
    streamed: bool,
    options: RunOptions<()>,
) -> Result<RunResult<()>> {
    if streamed {
        let mut s = Runner::run_streamed_with_options(agent, input, options);
        {
            let events = s.stream_events();
            futures::pin_mut!(events);
            while let Some(event) = events.next().await {
                event?;
            }
        }
        Ok(s.into_result().expect("a finished stream has a result"))
    } else {
        Runner::run_with_options(agent, input, options).await
    }
}

fn with_config(config: RunConfig<()>) -> RunOptions<()> {
    RunOptions::default().run_config(config)
}

/// `assert_redacted`: the secret reaches neither the model, the history, the traces nor the logs.
fn assert_redacted(secret: &str, result: &RunResult<()>, model: &ScriptedModel) {
    assert!(
        !serde_json::to_string(&last_input(model))
            .unwrap()
            .contains(secret)
    );
    let history: Vec<Value> = result.to_input_list().into_iter().map(|i| i.0).collect();
    assert!(!serde_json::to_string(&history).unwrap().contains(secret));
    let spans = spans();
    assert_ne!(spans, "[]", "spans were recorded");
    assert!(!spans.contains(secret), "secret leaked to spans: {spans}");
    let logs = logs();
    assert!(!logs.contains(secret), "secret leaked to logs: {logs}");
}

fn failing_tool(name: &str, message: String) -> FunctionTool<()> {
    FunctionTool::new_no_args(name, "fails", move |_ctx| {
        let message = message.clone();
        async move { Err::<String, _>(AgentsError::other(message)) }
    })
}

// ===========================================================================
// test_tool_exception_formatter.py
// ===========================================================================

#[tokio::test]
async fn run_formatter_returns_only_approved_feedback() {
    // `async_handler` is folded in: Rust tool bodies are always async and the formatter is sync.
    for streamed in [false, true] {
        let _f = fixture().await;
        let tool = failing_tool("fail", SECRET.to_string());
        let received: Received = Arc::default();
        let seen = received.clone();
        let config = RunConfig::<()>::default()
            .with_trace_include_sensitive_data(true)
            .with_tool_error_formatter(move |args| {
                seen.lock().unwrap().push((
                    args.kind,
                    args.tool_type.clone(),
                    args.tool_name.clone(),
                    args.call_id.clone(),
                    args.default_message.clone(),
                    args.error.map(|e| e.to_string()).unwrap_or_default(),
                ));
                Some(APPROVED.to_string())
            });
        let model = make_model("fail");
        let agent = Agent::<()>::new("test")
            .with_tool(tool)
            .with_model_instance(model.clone());
        let result = run(agent, "start", streamed, with_config(config))
            .await
            .unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("done"));
        assert_eq!(model_output(&model), json!(APPROVED));
        let received = received.lock().unwrap().clone();
        assert_eq!(received.len(), 1);
        let (kind, tool_type, tool_name, call_id, default_message, error) = &received[0];
        assert_eq!(*kind, ToolErrorKind::ToolException);
        assert_eq!(tool_type, "function");
        assert_eq!(tool_name, "fail");
        assert_eq!(call_id, "call-5324");
        assert_eq!(default_message, GENERIC);
        assert!(
            error.contains(SECRET),
            "the formatter sees the original error"
        );
        assert_redacted(SECRET, &result, &model);
    }
}

#[tokio::test]
async fn formatter_fallback_does_not_expose_exception_or_bad_return() {
    // `response="none"` and `"empty"`; `"raise"` / `"invalid"` cannot exist (see report).
    for (response, expected) in [(None, GENERIC), (Some(""), "")] {
        let _f = fixture().await;
        let model = make_model("fail");
        let config = RunConfig::<()>::default()
            .with_trace_include_sensitive_data(true)
            .with_tool_error_formatter(move |_args| response.map(str::to_string));
        let agent = Agent::<()>::new("test")
            .with_tool(failing_tool("fail", SECRET.to_string()))
            .with_model_instance(model.clone());
        let result = run(agent, "start", false, with_config(config))
            .await
            .unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("done"));
        assert_eq!(model_output(&model), json!(expected));
        assert_redacted(SECRET, &result, &model);
    }
}

#[tokio::test]
async fn explicit_tool_policy_wins_over_run_formatter() {
    for policy in ["custom", "propagate", "explicit-default"] {
        let _f = fixture().await;
        let tool = failing_tool("fail", SECRET.to_string());
        let tool = match policy {
            "custom" => tool.with_failure_error_function(|_ctx, _e| "per-tool message".into()),
            "explicit-default" => tool.with_failure_error_function(default_tool_error_function),
            _ => tool.without_failure_error_function(),
        };
        let called = Arc::new(Mutex::new(0usize));
        let seen = called.clone();
        let config = RunConfig::<()>::default().with_tool_error_formatter(move |_args| {
            *seen.lock().unwrap() += 1;
            Some(APPROVED.to_string())
        });
        let model = make_model("fail");
        let agent = Agent::<()>::new("test")
            .with_tool(tool)
            .with_model_instance(model.clone());
        let outcome = run(agent, "start", false, with_config(config)).await;
        if policy == "propagate" {
            let err = outcome.unwrap_err();
            assert!(
                matches!(err, AgentsError::UserError { .. }),
                "expected a UserError, got {err:?}"
            );
            assert!(err.to_string().contains(SECRET), "{err}");
        } else {
            outcome.unwrap();
            let expected = if policy == "custom" {
                "per-tool message"
            } else {
                GENERIC
            };
            assert_eq!(model_output(&model), json!(expected), "policy={policy}");
        }
        assert_eq!(*called.lock().unwrap(), 0, "policy={policy}");
    }
}

#[tokio::test]
async fn timeout_keeps_existing_policy() {
    // The `cancel` half of the parametrization is Python-only (asyncio.CancelledError).
    let _f = fixture().await;
    let tool = FunctionTool::<()>::new_no_args("fail", "hangs", |_ctx| async {
        std::future::pending::<()>().await;
        Ok("unreachable".to_string())
    })
    .with_timeout(std::time::Duration::from_millis(10));
    let called = Arc::new(Mutex::new(0usize));
    let seen = called.clone();
    let config = RunConfig::<()>::default().with_tool_error_formatter(move |_args| {
        *seen.lock().unwrap() += 1;
        Some(APPROVED.to_string())
    });
    let model = make_model("fail");
    let agent = Agent::<()>::new("test")
        .with_tool(tool)
        .with_model_instance(model.clone());
    run(agent, "start", false, with_config(config))
        .await
        .unwrap();
    assert_eq!(*called.lock().unwrap(), 0);
    assert_eq!(
        model_output(&model),
        json!("Tool 'fail' timed out after 0.01 seconds.")
    );
}

#[tokio::test]
async fn concurrent_runs_keep_their_own_formatter_and_parent_cancellation() {
    let _f = fixture().await;
    let entered = Arc::new(tokio::sync::Notify::new());
    let first_model = make_model("fail");
    let second_model = make_model("fail");
    // The first run blocks inside its tool (the Rust formatter is synchronous, so the block
    // sits one step earlier) until the second run has finished.
    let gate = entered.clone();
    let blocking = FunctionTool::<()>::new_no_args("fail", "fails", move |_ctx| {
        let gate = gate.clone();
        async move {
            gate.notify_one();
            std::future::pending::<()>().await;
            Err::<String, _>(AgentsError::other(SECRET))
        }
    });
    let first_agent = Agent::<()>::new("first")
        .with_tool(blocking)
        .with_model_instance(first_model.clone());
    let first = tokio::spawn(async move {
        Runner::run_with_options(
            first_agent,
            "start",
            with_config(
                RunConfig::default().with_tool_error_formatter(|_a| Some("first run".into())),
            ),
        )
        .await
    });
    entered.notified().await;
    let second = Runner::run_with_options(
        Agent::<()>::new("second")
            .with_tool(failing_tool("fail", SECRET.to_string()))
            .with_model_instance(second_model.clone()),
        "start",
        with_config(RunConfig::default().with_tool_error_formatter(|_a| Some("second run".into()))),
    )
    .await
    .unwrap();
    assert_eq!(second.final_output_text().as_deref(), Some("done"));
    assert_eq!(model_output(&second_model), json!("second run"));
    first.abort();
    assert!(first.await.unwrap_err().is_cancelled());
    assert_eq!(first_model.calls().len(), 1);
}

#[tokio::test]
async fn schema_backed_tool_retains_program_failure_policy() {
    for program_call in [false, true] {
        let _f = fixture().await;
        let tool = failing_tool("fail", SECRET.to_string())
            .with_allowed_callers(vec!["direct".into(), "programmatic".into()])
            .with_output_json_schema(json!({
                "type": "object",
                "properties": {"ok": {"type": "boolean"}},
                "required": ["ok"],
                "additionalProperties": false,
            }))
            .unwrap();
        let called = Arc::new(Mutex::new(0usize));
        let seen = called.clone();
        let config = RunConfig::<()>::default().with_tool_error_formatter(move |_args| {
            *seen.lock().unwrap() += 1;
            Some(APPROVED.to_string())
        });
        let mut fc = json!({
            "type": "function_call", "id": "call", "call_id": "call", "name": "fail",
            "arguments": "{}",
        });
        let mut first = Vec::new();
        if program_call {
            fc["caller"] = json!({"type": "program", "caller_id": "synthetic-program"});
            first.push(OutputItem::from_value(json!({
                "type": "program", "id": "program-item", "call_id": "synthetic-program",
                "code": "fail()", "fingerprint": "synthetic-fingerprint",
            })));
        }
        first.push(OutputItem::from_value(fc));
        let model = model(vec![first, vec![assistant_message("done")]]);
        let agent = Agent::<()>::new("test")
            .with_tool(tool)
            .with_tool(openai_agents_sdk::tool::Tool::ProgrammaticToolCalling(
                openai_agents_sdk::tool::ProgrammaticToolCallingTool,
            ))
            .with_model_instance(model.clone());
        let outcome = run(agent, "start", false, with_config(config)).await;
        if program_call {
            let err = outcome.unwrap_err();
            assert!(matches!(err, AgentsError::UserError { .. }), "{err:?}");
            assert!(err.to_string().contains(SECRET), "{err}");
            assert_eq!(*called.lock().unwrap(), 0);
        } else {
            outcome.unwrap();
            assert_eq!(*called.lock().unwrap(), 1);
            assert_eq!(model_output(&model), json!(APPROVED));
        }
    }
}

#[tokio::test]
async fn agent_tool_retains_factory_failure_policy() {
    for policy in ["omitted", "custom", "explicit-default", "propagate"] {
        let _f = fixture().await;
        let child = Agent::<()>::new("child")
            .with_model_instance(model(vec![vec![assistant_message("child done")]]));
        let mut options = AgentAsToolOptions::<()>::new()
            .tool_name("child")
            .tool_description("Run child")
            .custom_output_extractor(|_r| async { Err(AgentsError::other(SECRET)) });
        options = match policy {
            "custom" => options.failure_error_function(|_ctx, _e| "per-tool".into()),
            "explicit-default" => options.failure_error_function(default_tool_error_function),
            _ => options,
        };
        let mut child_tool = child.as_tool_with(options);
        if policy == "propagate" {
            child_tool = child_tool.without_failure_error_function();
        }
        let model = model(vec![
            vec![call("child", r#"{"input":"task"}"#, "call-1")],
            vec![assistant_message("done")],
        ]);
        let called = Arc::new(Mutex::new(Vec::<String>::new()));
        let seen = called.clone();
        let config = RunConfig::<()>::default()
            .with_trace_include_sensitive_data(true)
            .with_tool_error_formatter(move |args| {
                seen.lock().unwrap().push(args.tool_name.clone());
                Some(APPROVED.to_string())
            });
        let agent = Agent::<()>::new("parent")
            .with_tool(child_tool)
            .with_model_instance(model.clone());
        let outcome = run(agent, "start", false, with_config(config)).await;
        if policy == "propagate" {
            let err = outcome.unwrap_err();
            assert!(matches!(err, AgentsError::UserError { .. }), "{err:?}");
            assert!(err.to_string().contains(SECRET), "{err}");
        } else {
            let result = outcome.unwrap();
            let expected = if policy == "custom" {
                "per-tool"
            } else {
                GENERIC
            };
            assert_eq!(model_output(&model), json!(expected), "policy={policy}");
            if policy == "omitted" {
                assert_redacted(SECRET, &result, &model);
            }
        }
        assert!(called.lock().unwrap().is_empty(), "policy={policy}");
    }
}

// --------------------------------------------------------------- MCP fake

#[derive(Debug)]
struct FailingServer {
    config: MCPServerConfig,
}

#[async_trait::async_trait]
impl MCPServer for FailingServer {
    fn name(&self) -> &str {
        "fake_mcp_server"
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
        Ok(vec![MCPTool {
            meta: None,
            name: "lookup".into(),
            title: None,
            description: None,
            input_schema: json!({}),
            annotations: None,
        }])
    }
    async fn call_tool(&self, _name: &str, _args: Option<Map<String, Value>>) -> Result<Value> {
        Err(AgentsError::other(SECRET))
    }
    async fn list_prompts(&self) -> Result<Vec<Value>> {
        Ok(vec![])
    }
    async fn get_prompt(&self, _n: &str, _a: Option<BTreeMap<String, String>>) -> Result<Value> {
        Ok(Value::Null)
    }
}

#[tokio::test]
async fn mcp_tool_retains_factory_failure_policy() {
    let cases = [
        ("agent", "omitted"),
        ("agent", "custom"),
        ("agent", "explicit-default"),
        ("agent", "propagate"),
        ("server", "custom"),
        ("server", "explicit-default"),
        ("server", "propagate"),
    ];
    for (level, policy) in cases {
        let _f = fixture().await;
        let handler: Option<openai_agents_sdk::tool::ToolErrorFn<()>> = match policy {
            "custom" => Some(openai_agents_sdk::tool::tool_error_fn(
                |_c: &openai_agents_sdk::RunContext<()>, _e: &AgentsError| "per-tool".to_string(),
            )),
            "explicit-default" => Some(openai_agents_sdk::tool::tool_error_fn(
                default_tool_error_function::<()>,
            )),
            _ => None,
        };
        let mut server_config = MCPServerConfig::default();
        let mut mcp_config = MCPConfig::<()>::default();
        if level == "server" {
            server_config.tool_policy = Some(Arc::new(
                MCPToolPolicy::<()>::new().with_failure_error_function(handler),
            ));
            // The server policy wins over an agent-wide MCP policy as well as the formatter.
            mcp_config = mcp_config.with_failure_error_function(|_c, _e| "agent policy".into());
        } else if policy != "omitted" {
            mcp_config.failure_error_function = handler;
        }
        let server: Arc<dyn MCPServer> = Arc::new(FailingServer {
            config: server_config,
        });
        let model = make_model("lookup");
        let agent = Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_mcp_server(server)
            .with_mcp_config(mcp_config);
        let called = Arc::new(Mutex::new(0usize));
        let seen = called.clone();
        let config = RunConfig::<()>::default()
            .with_trace_include_sensitive_data(true)
            .with_tool_error_formatter(move |_args| {
                *seen.lock().unwrap() += 1;
                Some(APPROVED.to_string())
            });
        let outcome = run(agent, "start", false, with_config(config)).await;
        if policy == "propagate" {
            let err = outcome.unwrap_err();
            assert!(err.to_string().contains(SECRET), "{level}/{policy}: {err}");
        } else {
            let result = outcome.unwrap();
            let expected = if policy == "custom" {
                "per-tool"
            } else {
                GENERIC
            };
            assert_eq!(model_output(&model), json!(expected), "{level}/{policy}");
            if policy == "omitted" {
                assert_redacted(SECRET, &result, &model);
            }
        }
        assert_eq!(*called.lock().unwrap(), 0, "{level}/{policy}");
    }
}

#[tokio::test]
async fn run_formatter_uses_qualified_tool_identity() {
    let _f = fixture().await;
    let lookup = || failing_tool("lookup", SECRET.to_string());
    let mut tools = tool_namespace("crm", "CRM", vec![lookup()]).unwrap();
    tools.extend(tool_namespace("billing", "Billing", vec![lookup()]).unwrap());
    let model = model(vec![
        vec![
            function_call_full("lookup", "{}", "crm-call", None, Some("crm".into())),
            function_call_full("lookup", "{}", "billing-call", None, Some("billing".into())),
        ],
        vec![assistant_message("done")],
    ]);
    let mut agent = Agent::<()>::new("test").with_model_instance(model.clone());
    for t in tools {
        agent = agent.with_tool(t);
    }
    let config = RunConfig::<()>::default().with_tool_error_formatter(|args| {
        (args.tool_name == "crm.lookup").then(|| APPROVED.to_string())
    });
    run(agent, "start", false, with_config(config))
        .await
        .unwrap();
    let outputs: BTreeMap<String, Value> = last_input(&model)
        .into_iter()
        .filter(|i| i["type"] == "function_call_output")
        .map(|i| {
            (
                i["call_id"].as_str().unwrap().to_string(),
                i["output"].clone(),
            )
        })
        .collect();
    assert_eq!(
        outputs,
        BTreeMap::from([
            ("crm-call".to_string(), json!(APPROVED)),
            ("billing-call".to_string(), json!(GENERIC)),
        ])
    );
}

#[tokio::test]
async fn run_formatter_collapses_deferred_top_level_namespace() {
    let _f = fixture().await;
    let tool = failing_tool("lookup", SECRET.to_string()).with_defer_loading(true);
    let model = model(vec![
        vec![function_call_full(
            "lookup",
            "{}",
            "call-1",
            None,
            Some("lookup".into()),
        )],
        vec![assistant_message("done")],
    ]);
    let config = RunConfig::<()>::default().with_tool_error_formatter(|args| {
        (args.tool_name == "lookup").then(|| APPROVED.to_string())
    });
    let agent = Agent::<()>::new("test")
        .with_tool(tool)
        .with_tool(openai_agents_sdk::tool::Tool::ToolSearch(
            openai_agents_sdk::tool::ToolSearchTool::default(),
        ))
        .with_model_instance(model.clone());
    run(agent, "start", false, with_config(config))
        .await
        .unwrap();
    assert_eq!(model_output(&model), json!(APPROVED));
}

// ===========================================================================
// test_default_tool_error_redaction.py
// ===========================================================================

#[tokio::test]
async fn default_failure_keeps_exception_out_of_model_history_and_traces() {
    // `sync` and `async` failures are the same thing in Rust; `cancel` is Python-only.
    for streamed in [false, true] {
        for sensitive_tracing in [false, true] {
            let _f = fixture().await;
            let tool = failing_tool("raise_error", SENTINEL.to_string());
            let model = model(vec![
                vec![call("raise_error", "{}", "failure-call")],
                vec![assistant_message("recovered")],
            ]);
            let agent = Agent::<()>::new("test")
                .with_tool(tool)
                .with_model_instance(model.clone());
            let config =
                RunConfig::<()>::default().with_trace_include_sensitive_data(sensitive_tracing);
            let result = run(agent, "start", streamed, with_config(config))
                .await
                .unwrap();
            assert_eq!(result.final_output_text().as_deref(), Some("recovered"));
            let outputs: Vec<Value> = last_input(&model)
                .into_iter()
                .filter(|i| i["type"] == "function_call_output")
                .collect();
            assert_eq!(outputs.len(), 1);
            assert_eq!(outputs[0]["output"], json!(GENERIC));
            assert_redacted(SENTINEL, &result, &model);
        }
    }
}

#[tokio::test]
async fn explicit_failure_policies_remain_available() {
    for policy in ["custom", "propagate", "explicit-default"] {
        let _f = fixture().await;
        let received = Arc::new(Mutex::new(Vec::<String>::new()));
        let seen = received.clone();
        let tool = failing_tool("failing_tool", SENTINEL.to_string());
        let tool = match policy {
            "custom" => tool.with_failure_error_function(move |_ctx, e| {
                seen.lock().unwrap().push(e.to_string());
                "Application-approved feedback".into()
            }),
            "explicit-default" => tool.with_failure_error_function(default_tool_error_function),
            _ => tool.without_failure_error_function(),
        };
        let model = model(vec![
            vec![call("failing_tool", "{}", "call-1")],
            vec![assistant_message("recovered")],
        ]);
        let agent = Agent::<()>::new("test")
            .with_tool(tool)
            .with_model_instance(model.clone());
        let outcome = run(agent, "start", false, RunOptions::default()).await;
        if policy == "propagate" {
            let err = outcome.unwrap_err();
            assert!(matches!(err, AgentsError::UserError { .. }), "{err:?}");
            assert!(err.to_string().contains(SENTINEL), "{err}");
            continue;
        }
        let result = outcome.unwrap();
        assert_eq!(result.final_output_text().as_deref(), Some("recovered"));
        if policy == "custom" {
            assert_eq!(model_output(&model), json!("Application-approved feedback"));
            assert_eq!(*received.lock().unwrap(), vec![SENTINEL.to_string()]);
        } else {
            assert_eq!(model_output(&model), json!(GENERIC));
            assert!(!spans().contains(SENTINEL), "{}", spans());
        }
    }
}

#[tokio::test]
async fn explicit_local_diagnostics_keep_exception_detail() {
    let _f = fixture().await;
    set_dont_log_tool_data(false);
    let model = model(vec![
        vec![call("failing_tool", "{}", "call-1")],
        vec![assistant_message("recovered")],
    ]);
    let agent = Agent::<()>::new("test")
        .with_tool(failing_tool("failing_tool", SENTINEL.to_string()))
        .with_model_instance(model.clone());
    let result = run(agent, "start", false, RunOptions::default())
        .await
        .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("recovered"));
    assert!(logs().contains(SENTINEL), "logs: {}", logs());
    assert!(
        !serde_json::to_string(&last_input(&model))
            .unwrap()
            .contains(SENTINEL)
    );
    assert!(!spans().contains(SENTINEL));
}

// ===========================================================================
// test_model_error_logging_redaction.py
// ===========================================================================

#[tokio::test]
async fn model_error_annotation_respects_log_redaction() {
    // Rust spans cannot fail to record an error (Python's `_FailingSpan`), so the port checks
    // the remaining contract: recording a model error never logs the model data.
    let _f = fixture().await;
    set_dont_log_model_data(true);
    let span = openai_agents_sdk::tracing::create_span(
        openai_agents_sdk::tracing::SpanData::Custom {
            name: "model".into(),
            data: Value::Null,
        },
        None,
        false,
    );
    openai_agents_sdk::tracing::record_model_error_on_span(
        &span,
        "Error getting response",
        &"model-secret",
        false,
    );
    assert!(!logs().contains("model-secret"));
    let exported = serde_json::to_string(&span.export()).unwrap();
    assert!(!exported.contains("model-secret"), "{exported}");
}
