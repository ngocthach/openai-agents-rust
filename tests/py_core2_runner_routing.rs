//! Ports of the routing, tool-identity and run-configuration tests in
//! `tests/test_agent_runner.py` (Python lines ~1019-2460): hosted MCP sticky approvals, tool
//! name collisions, handoff enablement context, default runner registry, resumed-input
//! normalization, the function-tool queue, reasoning-id policy and pre-approval guardrails.
//!
//! Every test names the Python test(s) it ports in its doc comment. Python tests parametrized
//! on `streamed` iterate both modes inside one Rust test.

#![allow(clippy::field_reassign_with_default)]

use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::agent::AgentAsToolOptions;
use openai_agents_sdk::items::{OutputItem, RunInput, RunItemKind};
use openai_agents_sdk::lifecycle::RunHooks;
use openai_agents_sdk::memory::{InMemorySession, Session};
use openai_agents_sdk::result::RunResult;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::{
    ReasoningItemIdPolicy, RunConfig, ToolExecutionConfig, ToolNameCollisionPolicy,
};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::{FunctionTool, HostedMCPTool, Tool};
use openai_agents_sdk::tool_guardrails::{ToolGuardrailFunctionOutput, ToolInputGuardrail};
use openai_agents_sdk::{Agent, AgentsError, InputItem, RunContext, RunState, Runner, handoff};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    install_logger();
    Arc::new(FakeModel::new())
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct AnyArgs {
    #[serde(default)]
    a: String,
}

fn tool_returning<C: Send + Sync + 'static>(name: &str, value: &str) -> FunctionTool<C> {
    let value = value.to_string();
    FunctionTool::new(name, "a tool", move |_ctx, _a: AnyArgs| {
        let value = value.clone();
        async move { Ok(value) }
    })
}

fn reasoning(id: &str, summary: Option<&str>) -> OutputItem {
    let summary = match summary {
        Some(t) => json!([{"type": "summary_text", "text": t}]),
        None => json!([]),
    };
    OutputItem::from_value(json!({"type": "reasoning", "id": id, "summary": summary}))
}

fn handoff_call(target: &str) -> OutputItem {
    openai_agents_sdk::testing::function_call(format!("transfer_to_{target}"), "{}")
}

/// Run in either mode and return the final result (Python `_run_agent_with_optional_streaming`).
async fn run_mode<C: Send + Sync + 'static>(
    agent: &Arc<Agent<C>>,
    input: impl Into<RunInput>,
    options: RunOptions<C>,
    streamed: bool,
) -> Result<RunResult<C>, AgentsError> {
    if !streamed {
        return Runner::run_with_options(agent.clone(), input, options).await;
    }
    let mut stream = Runner::run_streamed_with_options(agent.clone(), input, options);
    let mut error = None;
    {
        let mut events = std::pin::pin!(stream.stream_events());
        while let Some(ev) = events.next().await {
            if let Err(e) = ev {
                error = Some(e);
            }
        }
    }
    match error {
        Some(e) => Err(e),
        None => Ok(stream
            .into_result()
            .expect("a finished stream has a result")),
    }
}

fn input_of(call: &openai_agents_sdk::models::interface::ModelRequest) -> Vec<Value> {
    call.input.iter().map(|i| i.0.clone()).collect()
}

fn find_reasoning(items: &[Value]) -> Option<Value> {
    items
        .iter()
        .find(|i| i.get("type").and_then(Value::as_str) == Some("reasoning"))
        .cloned()
}

// ------------------------------------------------------------------------ log capture

struct Capture;
static LOGS: OnceLock<Mutex<Vec<String>>> = OnceLock::new();
/// Serializes the tests that flip the global tool-data logging flag.
static LOG_FLAG: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

impl log::Log for Capture {
    fn enabled(&self, _m: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        if record.level() <= log::Level::Warn {
            LOGS.get_or_init(Default::default)
                .lock()
                .unwrap()
                .push(record.args().to_string());
        }
    }
    fn flush(&self) {}
}

fn install_logger() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let _ = log::set_boxed_logger(Box::new(Capture));
        log::set_max_level(log::LevelFilter::Warn);
    });
}

fn logs_containing(needle: &str) -> Vec<String> {
    LOGS.get_or_init(Default::default)
        .lock()
        .unwrap()
        .iter()
        .filter(|m| m.contains(needle))
        .cloned()
        .collect()
}

// ------------------------------------------------------------------ hosted MCP approvals

/// Python: `test_persistent_hosted_mcp_approval_does_not_cross_servers[streamed=False|True]`.
#[tokio::test]
async fn persistent_hosted_mcp_approval_does_not_cross_servers() {
    for streamed in [false, true] {
        let model = setup();
        let request = |id: &str, server: &str| {
            OutputItem::from_value(json!({
                "type": "mcp_approval_request", "id": id, "arguments": "{}",
                "name": "lookup_account", "server_label": server
            }))
        };
        model.add_turn(vec![request("request-a", "server-a")]);
        model.add_turn(vec![request("request-b", "server-b")]);
        let agent = Arc::new(
            Agent::<()>::new("test")
                .with_model_instance(model.clone())
                .with_tool(Tool::HostedMCP(HostedMCPTool::with_server(
                    "server-a",
                    "https://server-a.example/mcp",
                )))
                .with_tool(Tool::HostedMCP(HostedMCPTool::with_server(
                    "server-b",
                    "https://server-b.example/mcp",
                ))),
        );

        let first = run_mode(&agent, "hello", RunOptions::new(()), streamed)
            .await
            .unwrap();
        assert_eq!(first.interruptions.len(), 1, "streamed={streamed}");
        assert_eq!(
            first.interruptions[0].raw_item.extra.get("server_label"),
            Some(&json!("server-a"))
        );
        let mut state = first.to_state();
        state.approve(&first.interruptions[0], true);
        let restored = RunState::from_json(state.to_json().unwrap()).unwrap();

        let resumed = run_mode(&agent, restored, RunOptions::new(()), streamed)
            .await
            .unwrap();
        assert_eq!(resumed.interruptions.len(), 1, "streamed={streamed}");
        assert_eq!(
            resumed.interruptions[0].raw_item.extra.get("server_label"),
            Some(&json!("server-b")),
            "streamed={streamed}"
        );
    }
}

// --------------------------------------------------------------------- tool collisions

fn collision_agent(surface: &str, model: &Arc<FakeModel>, tag: &str) -> Arc<Agent<()>> {
    let billing = Agent::<()>::new(format!("Billing Agent {tag}"));
    let normalized = Agent::<()>::new(format!("billing agent {tag}"));
    let derived = |a: &Agent<()>, d: &str| {
        a.as_tool_with(AgentAsToolOptions::new().tool_description(d.to_string()))
    };
    let agent = Agent::<()>::new("triage").with_model_instance(model.clone());
    Arc::new(match surface {
        "agent_tool" => agent
            .with_tool(derived(&billing, "First billing agent"))
            .with_tool(derived(&normalized, "Second billing agent")),
        "handoff" => agent
            .with_handoff(handoff(Arc::new(billing)))
            .with_handoff(handoff(Arc::new(normalized))),
        _ => agent
            .with_tool(derived(
                &Agent::<()>::new(format!("transfer to Billing Agent {tag}")),
                "Billing tool",
            ))
            .with_handoff(handoff(Arc::new(billing))),
    })
}

/// Python: `test_run_reports_derived_agent_name_collisions_before_model_call`
/// (`surface` x `streamed` x `collision_policy`, 12 cases).
#[tokio::test]
async fn derived_agent_name_collisions_are_reported_before_the_model_call() {
    let _flag = LOG_FLAG.lock().await;
    openai_agents_sdk::debug::set_dont_log_tool_data(false);
    let mut case = 0;
    for surface in ["agent_tool", "handoff", "mixed"] {
        for streamed in [false, true] {
            for policy in [
                ToolNameCollisionPolicy::Warn,
                ToolNameCollisionPolicy::Error,
            ] {
                case += 1;
                let tag = format!("c{case}");
                let model = setup();
                model.add_turn(vec![assistant_message("done")]);
                let agent = collision_agent(surface, &model, &tag);
                let mut config = RunConfig::default();
                config.tool_name_collision_policy = policy;
                let result = run_mode(
                    &agent,
                    "Route this request",
                    RunOptions::new(()).run_config(config),
                    streamed,
                )
                .await;
                let label = format!("{surface} streamed={streamed} {policy:?}");
                if policy == ToolNameCollisionPolicy::Error {
                    let err = result.expect_err(&label).to_string();
                    let expected = match surface {
                        "agent_tool" => "Ambiguous agent tool configuration",
                        "handoff" => "Ambiguous handoff configuration",
                        _ => "Ambiguous agent routing configuration",
                    };
                    assert!(err.contains(expected), "{label}: {err}");
                    assert!(model.calls().is_empty(), "{label}");
                } else {
                    result.expect(&label);
                    assert!(!model.calls().is_empty(), "{label}");
                    let messages = logs_containing(&format!("{tag}'"));
                    let ambiguous: Vec<_> = messages
                        .iter()
                        .filter(|m| m.starts_with("Ambiguous "))
                        .collect();
                    assert!(!ambiguous.is_empty(), "{label}: {messages:?}");
                    assert!(ambiguous[0].contains("Pass an explicit"), "{label}");
                }
            }
        }
    }
}

fn lookup_tool(record: &Arc<Mutex<Vec<&'static str>>>, which: &'static str) -> FunctionTool<()> {
    let record = record.clone();
    FunctionTool::new("lookup", "lookup", move |_ctx, _a: AnyArgs| {
        let record = record.clone();
        async move {
            record.lock().unwrap().push(which);
            Ok(which.to_string())
        }
    })
}

const DUPLICATE_LOOKUP: &str = "Ambiguous function tool configuration: the tool name `lookup` is \
used by multiple tools. Assign a unique routed name to every colliding function tool with \
`name_override=`, `tool_name=`, or a namespace.";

/// Python: `test_run_warns_and_keeps_last_duplicate_function_tool[streamed=False|True]`.
#[tokio::test]
async fn a_duplicate_function_tool_warns_and_the_last_one_wins() {
    let _flag = LOG_FLAG.lock().await;
    openai_agents_sdk::debug::set_dont_log_tool_data(false);
    for streamed in [false, true] {
        let model = setup();
        model.add_turn(vec![function_call_with_id("lookup", "{}", "call-1")]);
        model.add_turn(vec![assistant_message("done")]);
        let calls = Arc::new(Mutex::new(Vec::new()));
        let agent = Arc::new(
            Agent::<()>::new("agent")
                .with_model_instance(model.clone())
                .with_tool(lookup_tool(&calls, "first"))
                .with_tool(lookup_tool(&calls, "second")),
        );
        run_mode(&agent, "Look this up", RunOptions::new(()), streamed)
            .await
            .unwrap();
        assert_eq!(
            *calls.lock().unwrap(),
            vec!["second"],
            "streamed={streamed}"
        );
        let first = &model.calls()[0];
        assert_eq!(first.tools.len(), 1);
        assert!(
            logs_containing(DUPLICATE_LOOKUP)
                .iter()
                .all(|m| m == DUPLICATE_LOOKUP)
                && !logs_containing(DUPLICATE_LOOKUP).is_empty()
        );
    }
}

/// Python: `test_collision_warning_redacts_tool_data`,
/// `test_collision_warning_preserves_tool_diagnostics_when_enabled`.
#[tokio::test]
async fn the_collision_warning_honors_the_tool_data_logging_policy() {
    let _flag = LOG_FLAG.lock().await;
    for (redact, name) in [
        (true, "tenant_secret_tool_token"),
        (false, "diagnostic_tool_name"),
    ] {
        openai_agents_sdk::debug::set_dont_log_tool_data(redact);
        let model = setup();
        model.add_turn(vec![assistant_message("done")]);
        let agent = Arc::new(
            Agent::<()>::new("agent")
                .with_model_instance(model.clone())
                .with_tool(tool_returning::<()>(name, "first"))
                .with_tool(tool_returning::<()>(name, "second")),
        );
        let before = logs_containing("collision detected").len();
        run_mode(&agent, "go", RunOptions::new(()), false)
            .await
            .unwrap();
        if redact {
            assert!(logs_containing(name).is_empty());
            let redacted = logs_containing(
                "Tool name collision detected. Assign unique routed tool names or enable tool \
                 data logging for details.",
            );
            assert!(redacted.len() > before);
        } else {
            assert!(!logs_containing(name).is_empty());
        }
    }
    openai_agents_sdk::debug::set_dont_log_tool_data(false);
}

/// Python: `test_run_rejects_duplicate_function_tools_in_error_mode[streamed=False|True]`.
#[tokio::test]
async fn duplicate_function_tools_are_rejected_in_error_mode() {
    for streamed in [false, true] {
        let model = setup();
        model.add_turn(vec![assistant_message("done")]);
        let calls = Arc::new(Mutex::new(Vec::new()));
        let agent = Arc::new(
            Agent::<()>::new("agent")
                .with_model_instance(model.clone())
                .with_tool(lookup_tool(&calls, "first"))
                .with_tool(lookup_tool(&calls, "second")),
        );
        let mut config = RunConfig::default();
        config.tool_name_collision_policy = ToolNameCollisionPolicy::Error;
        let err = run_mode(
            &agent,
            "Look this up",
            RunOptions::new(()).run_config(config),
            streamed,
        )
        .await
        .unwrap_err();
        assert!(matches!(err, AgentsError::UserError { .. }), "{err:?}");
        assert!(
            err.to_string()
                .contains("the tool name `lookup` is used by multiple tools"),
            "{err}"
        );
        assert!(model.calls().is_empty());
    }
}

/// Python: `test_run_warns_once_for_repeated_source_agent_name`.
#[tokio::test]
async fn a_repeated_source_agent_name_warns_once() {
    let _flag = LOG_FLAG.lock().await;
    openai_agents_sdk::debug::set_dont_log_tool_data(false);
    let model = setup();
    model.add_turn(vec![assistant_message("done")]);
    let derived = |name: &str, d: &str| {
        Agent::<()>::new(name)
            .as_tool_with(AgentAsToolOptions::new().tool_description(d.to_string()))
    };
    let agent = Arc::new(
        Agent::<()>::new("orchestrator")
            .with_model_instance(model.clone())
            .with_tool(derived("Refund", "First refund agent"))
            .with_tool(derived("refund", "Second refund agent"))
            .with_tool(derived("refund", "Third refund agent")),
    );
    Runner::run_with_options(agent.clone(), "Route this request", RunOptions::new(()))
        .await
        .unwrap();
    let expected = "Ambiguous function tool configuration: the tool name `refund` is used by \
                    multiple tools. Assign a unique routed name to every colliding function \
                    tool with `name_override=`, `tool_name=`, or a namespace.";
    assert_eq!(logs_containing("`refund`"), vec![expected.to_string()]);
    let tools = &model.calls()[0].tools;
    assert_eq!(tools.len(), 1);
    assert!(format!("{:?}", tools[0]).contains("Third refund agent"));
}

/// Python: `test_multiway_mixed_collision_reports_every_owner_must_be_unique`.
#[tokio::test]
async fn a_multiway_mixed_collision_keeps_the_handoff() {
    let _flag = LOG_FLAG.lock().await;
    openai_agents_sdk::debug::set_dont_log_tool_data(false);
    let model = setup();
    model.add_turn(vec![assistant_message("done")]);
    let agent = Arc::new(
        Agent::<()>::new("router")
            .with_model_instance(model.clone())
            .with_tool(tool_returning::<()>("route", "first"))
            .with_tool(tool_returning::<()>("route", "second"))
            .with_handoff(handoff(Arc::new(Agent::<()>::new("Billing"))).with_tool_name("route")),
    );
    Runner::run_with_options(agent.clone(), "go", RunOptions::new(()))
        .await
        .unwrap();
    let call = &model.calls()[0];
    assert!(call.tools.is_empty(), "{:?}", call.tools);
    assert_eq!(call.handoffs.len(), 1);
    assert_eq!(
        logs_containing("`route`"),
        vec![
            "Ambiguous tool routing configuration: the tool name `route` is used by both a \
             function tool and a handoff. Assign a unique routed name to every colliding \
             function tool and handoff with `name_override=`, `tool_name=`, \
             `tool_name_override=`, or a namespace."
                .to_string()
        ]
    );
}

// ----------------------------------------------------------------- handoff enablement

struct InitializeContextHooks;

#[async_trait]
impl RunHooks<Mutex<HashMap<String, bool>>> for InitializeContextHooks {
    async fn on_agent_start(
        &self,
        context: &RunContext<Mutex<HashMap<String, bool>>>,
        agent: &Agent<Mutex<HashMap<String, bool>>>,
    ) {
        if agent.name == "source" {
            context
                .context()
                .lock()
                .unwrap()
                .insert("hook_initialized".into(), true);
        }
    }
}

/// Python: `test_handoff_enablement_uses_initialized_turn_context[streamed=False|True]`.
#[tokio::test]
async fn handoff_enablement_sees_the_initialized_turn_context() {
    type Ctx = Mutex<HashMap<String, bool>>;
    for streamed in [false, true] {
        let model = setup();
        let target = Arc::new(Agent::<Ctx>::new("target").with_model_instance(model.clone()));
        model.add_turn(vec![handoff_call("target")]);
        model.add_turn(vec![assistant_message("done")]);
        let observed = Arc::new(Mutex::new(Vec::new()));
        let seen = observed.clone();
        let source = Arc::new(
            Agent::<Ctx>::new("source")
                .with_model_instance(model.clone())
                .with_dynamic_prompt(|ctx: RunContext<Ctx>, _agent| async move {
                    ctx.context()
                        .lock()
                        .unwrap()
                        .insert("prompt_initialized".into(), true);
                    Ok(Some(openai_agents_sdk::prompts::Prompt::new("prompt-id")))
                })
                .with_handoff(handoff(target).with_is_enabled_fn(
                    move |ctx: RunContext<Ctx>, agent: String| {
                        let seen = seen.clone();
                        async move {
                            let turn: Vec<Value> =
                                ctx.turn_input().into_iter().map(|i| i.0).collect();
                            let snapshot = ctx.context().lock().unwrap().clone();
                            seen.lock().unwrap().push((turn.clone(), snapshot.clone()));
                            agent == "source"
                                && turn == vec![json!({"content": "current turn", "role": "user"})]
                                && snapshot.get("hook_initialized") == Some(&true)
                                && snapshot.get("prompt_initialized") == Some(&true)
                        }
                    },
                )),
        );
        let options =
            RunOptions::new(Mutex::new(HashMap::new())).hooks(Arc::new(InitializeContextHooks));
        let outcome = run_mode(&source, "current turn", options, streamed).await;
        let observed = observed.lock().unwrap().clone();
        assert!(outcome.is_ok(), "{observed:?}");
        let expected: HashMap<String, bool> = [
            ("hook_initialized".to_string(), true),
            ("prompt_initialized".to_string(), true),
        ]
        .into();
        assert_eq!(
            observed,
            vec![(
                vec![json!({"content": "current turn", "role": "user"})],
                expected
            )],
            "streamed={streamed}"
        );
    }
}

// ------------------------------------------------------------- default runner registry

/// Python: `test_set_default_agent_runner_roundtrip`.
#[test]
fn set_default_agent_runner_round_trips() {
    use openai_agents_sdk::run::{
        AgentRunner, DefaultAgentRunner, get_default_agent_runner, set_default_agent_runner,
    };
    struct Marker;
    let runner: Arc<dyn AgentRunner<Marker>> = Arc::new(DefaultAgentRunner);
    set_default_agent_runner(Some(runner.clone()));
    let got = get_default_agent_runner::<Marker>().expect("registered");
    assert!(Arc::ptr_eq(&got, &runner));
    set_default_agent_runner::<Marker>(None);
    assert!(get_default_agent_runner::<Marker>().is_none());
}

/// Python: `test_run_config_defaults_nested_handoff_history_opt_in`.
#[test]
fn nested_handoff_history_is_opt_in() {
    assert!(!RunConfig::<()>::default().nest_handoff_history);
}

// ------------------------------------------------------------- resumed-input normalization

fn resumed_state_with_input(agent: &str, input: Vec<Value>) -> RunState {
    RunState {
        schema_version: openai_agents_sdk::run_state::CURRENT_SCHEMA_VERSION.to_string(),
        current_agent_name: agent.into(),
        original_input: input.into_iter().map(InputItem).collect(),
        max_turns: Some(10),
        ..Default::default()
    }
}

async fn model_input_after_resume(input: Vec<Value>) -> Vec<Value> {
    let model = setup();
    model.add_turn(vec![assistant_message("done")]);
    let agent = Arc::new(Agent::<()>::new("test").with_model_instance(model.clone()));
    let state = resumed_state_with_input("test", input);
    Runner::run_with_options(agent, state, RunOptions::new(()))
        .await
        .unwrap();
    input_of(&model.calls()[0])
}

/// Python: `test_normalize_resumed_input_drops_orphan_function_calls`.
#[tokio::test]
async fn a_resumed_input_drops_orphan_function_calls() {
    let sent = model_input_after_resume(vec![
        json!({"type": "function_call", "call_id": "orphan_call", "name": "tool_orphan", "arguments": "{}"}),
        json!({"type": "function_call", "call_id": "paired_call", "name": "tool_paired", "arguments": "{}"}),
        json!({"type": "function_call_output", "call_id": "paired_call", "output": "ok"}),
    ])
    .await;
    let ids: Vec<&str> = sent
        .iter()
        .filter(|i| i["type"] == "function_call")
        .filter_map(|i| i["call_id"].as_str())
        .collect();
    assert_eq!(ids, vec!["paired_call"]);
}

/// Python: `testdrop_orphan_function_calls_removes_orphans` (every call family).
#[tokio::test]
async fn a_resumed_input_drops_orphans_of_every_call_family() {
    let sent = model_input_after_resume(vec![
        json!({"type": "function_call", "call_id": "call_orphan", "name": "tool_one", "arguments": "{}"}),
        json!({"type": "message", "role": "user", "content": "hello"}),
        json!({"type": "function_call", "call_id": "call_keep", "name": "tool_keep", "arguments": "{}"}),
        json!({"type": "function_call_output", "call_id": "call_keep", "output": "done"}),
        json!({"type": "shell_call", "call_id": "shell_orphan"}),
        json!({"type": "shell_call", "call_id": "shell_keep"}),
        json!({"type": "shell_call_output", "call_id": "shell_keep", "output": []}),
        json!({"type": "apply_patch_call", "call_id": "patch_orphan"}),
        json!({"type": "apply_patch_call", "call_id": "patch_keep"}),
        json!({"type": "apply_patch_call_output", "call_id": "patch_keep", "output": "done"}),
        json!({"type": "computer_call", "call_id": "computer_orphan"}),
        json!({"type": "computer_call", "call_id": "computer_keep"}),
        json!({"type": "computer_call_output", "call_id": "computer_keep", "output": {}}),
        json!({"type": "local_shell_call", "call_id": "local_shell_orphan"}),
        json!({"type": "local_shell_call", "call_id": "local_shell_keep"}),
        json!({"type": "local_shell_call_output", "call_id": "local_shell_keep",
               "output": {"stdout": "", "stderr": "", "outcome": {}}}),
    ])
    .await;
    for orphan in [
        "call_orphan",
        "shell_orphan",
        "patch_orphan",
        "computer_orphan",
        "local_shell_orphan",
    ] {
        assert!(
            !sent.iter().any(|i| i["call_id"] == orphan),
            "{orphan} survived: {sent:?}"
        );
    }
    for (ty, id) in [
        ("function_call", "call_keep"),
        ("shell_call", "shell_keep"),
        ("apply_patch_call", "patch_keep"),
        ("computer_call", "computer_keep"),
        ("local_shell_call", "local_shell_keep"),
    ] {
        assert!(
            sent.iter().any(|i| i["type"] == ty && i["call_id"] == id),
            "{ty} {id} missing"
        );
    }
}

/// Python: `test_normalize_resumed_input_drops_orphan_tool_search_calls`.
#[tokio::test]
async fn a_resumed_input_drops_orphan_tool_search_calls() {
    let sent = model_input_after_resume(vec![
        json!({"type": "tool_search_call", "call_id": "orphan_search", "arguments": {"query": "orphan"},
               "execution": "server", "status": "completed"}),
        json!({"type": "tool_search_call", "call_id": "paired_search", "arguments": {"query": "paired"},
               "execution": "server", "status": "completed"}),
        json!({"type": "tool_search_output", "call_id": "paired_search", "execution": "server",
               "status": "completed", "tools": []}),
    ])
    .await;
    let ids: Vec<&str> = sent
        .iter()
        .filter(|i| i["type"] == "tool_search_call")
        .filter_map(|i| i["call_id"].as_str())
        .collect();
    assert_eq!(ids, vec!["paired_search"]);
}

/// Python: `test_normalize_resumed_input_preserves_hosted_tool_search_pair_without_call_ids`.
#[tokio::test]
async fn a_resumed_input_keeps_an_anonymous_hosted_tool_search_pair() {
    let sent = model_input_after_resume(vec![
        json!({"type": "tool_search_call", "call_id": null, "arguments": {"query": "paired"},
               "execution": "server", "status": "completed"}),
        json!({"type": "tool_search_output", "call_id": null, "execution": "server",
               "status": "completed", "tools": []}),
    ])
    .await;
    let types: Vec<&str> = sent.iter().filter_map(|i| i["type"].as_str()).collect();
    assert_eq!(types, vec!["tool_search_call", "tool_search_output"]);
}

/// Python: `test_normalize_resumed_input_matches_latest_anonymous_tool_search_call`.
#[tokio::test]
async fn a_resumed_input_pairs_the_latest_anonymous_tool_search_call() {
    let sent = model_input_after_resume(vec![
        json!({"type": "tool_search_call", "call_id": null, "arguments": {"query": "orphan"},
               "execution": "server", "status": "completed"}),
        json!({"type": "tool_search_call", "call_id": null, "arguments": {"query": "paired"},
               "execution": "server", "status": "completed"}),
        json!({"type": "tool_search_output", "call_id": null, "execution": "server",
               "status": "completed", "tools": []}),
    ])
    .await;
    let types: Vec<&str> = sent.iter().filter_map(|i| i["type"].as_str()).collect();
    assert_eq!(types, vec!["tool_search_call", "tool_search_output"]);
    assert_eq!(sent[0]["arguments"], json!({"query": "paired"}));
}

/// Python: `testnormalize_input_items_for_api_preserves_provider_data`.
#[tokio::test]
async fn provider_data_on_input_items_reaches_the_model() {
    let sent = model_input_after_resume(vec![
        json!({"type": "function_call", "call_id": "call_norm", "name": "t", "arguments": "{}"}),
        json!({"type": "function_call_output", "call_id": "call_norm", "status": "completed",
               "output": "out", "provider_data": {"trace": "keep"}}),
        json!({"type": "message", "role": "user", "content": "hi", "provider_data": {"trace": "remove"}}),
    ])
    .await;
    assert_eq!(sent[1]["type"], "function_call_output");
    assert_eq!(sent[1]["call_id"], "call_norm");
    assert_eq!(sent[1]["provider_data"], json!({"trace": "keep"}));
    assert_eq!(sent[2]["role"], "user");
    assert_eq!(sent[2]["provider_data"], json!({"trace": "remove"}));
}

/// Python: `test_server_conversation_tracker_tracks_previous_response_id`.
#[tokio::test]
async fn the_server_tracker_chains_the_latest_response_id() {
    let model = setup();
    model.add_turn(vec![function_call_with_id("foo", "{}", "c1")]);
    model.add_turn(vec![assistant_message("hello")]);
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(tool_returning::<()>("foo", "r")),
    );
    let result = Runner::run_with_options(
        agent,
        "hi",
        RunOptions::new(()).previous_response_id("resp_a"),
    )
    .await
    .unwrap();
    let calls = model.calls();
    assert_eq!(calls[0].previous_response_id.as_deref(), Some("resp_a"));
    assert_eq!(
        calls[1].previous_response_id.as_deref(),
        Some("resp_fake_1")
    );
    // Only the new tool output is sent once the server holds the first response.
    assert_eq!(
        input_of(&calls[1])
            .iter()
            .map(|i| i["type"].as_str().unwrap_or("message").to_string())
            .collect::<Vec<_>>(),
        vec!["function_call_output"]
    );
    assert_eq!(result.last_response_id().as_deref(), Some("resp_fake_2"));
}

// ----------------------------------------------------------------- function tool queue

#[derive(Deserialize, JsonSchema)]
struct ValueArg {
    value: usize,
}

/// Python: `test_function_tool_queue_preserves_fifo_across_multiple_refills`.
#[tokio::test]
async fn the_function_tool_queue_is_fifo_across_refills() {
    use tokio::sync::Notify;
    let model = setup();
    model.add_turn(
        (0..7)
            .map(|v| {
                function_call_with_id(
                    "tracked_tool",
                    json!({"value": v}).to_string(),
                    v.to_string(),
                )
            })
            .collect(),
    );
    model.add_turn(vec![assistant_message("done")]);
    let started = Arc::new(Mutex::new(Vec::<usize>::new()));
    let completed = Arc::new(Mutex::new(Vec::<usize>::new()));
    let started_n: Arc<Vec<Notify>> = Arc::new((0..7).map(|_| Notify::new()).collect());
    let release: Arc<Vec<Notify>> = Arc::new((0..7).map(|_| Notify::new()).collect());
    let (s, c, sn, r) = (
        started.clone(),
        completed.clone(),
        started_n.clone(),
        release.clone(),
    );
    let tool = FunctionTool::<()>::new("tracked_tool", "t", move |_ctx, a: ValueArg| {
        let (s, c, sn, r) = (s.clone(), c.clone(), sn.clone(), r.clone());
        async move {
            s.lock().unwrap().push(a.value);
            sn[a.value].notify_one();
            r[a.value].notified().await;
            c.lock().unwrap().push(a.value);
            Ok(format!("ok-{}", a.value))
        }
    });
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(tool),
    );
    let mut config = RunConfig::default();
    config.tool_execution = ToolExecutionConfig {
        max_function_tool_concurrency: Some(2),
        ..Default::default()
    };
    let task = tokio::spawn(Runner::run_with_options(
        agent,
        "run tools",
        RunOptions::new(()).run_config(config),
    ));
    let wait = |n: usize| {
        let sn = started_n.clone();
        async move {
            tokio::time::timeout(std::time::Duration::from_secs(5), sn[n].notified())
                .await
                .expect("tool start");
        }
    };
    wait(1).await;
    assert_eq!(*started.lock().unwrap(), vec![0, 1]);
    for value in 1..6 {
        release[value].notify_one();
        wait(value + 1).await;
        assert_eq!(*started.lock().unwrap(), (0..value + 2).collect::<Vec<_>>());
        assert_eq!(
            *completed.lock().unwrap(),
            (1..value + 1).collect::<Vec<_>>()
        );
    }
    release[6].notify_one();
    release[0].notify_one();
    let result = tokio::time::timeout(std::time::Duration::from_secs(5), task)
        .await
        .unwrap()
        .unwrap()
        .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    let outputs: Vec<Value> = result
        .new_items
        .iter()
        .filter_map(|i| match &i.kind {
            RunItemKind::ToolCallOutput { output, .. } => Some(output.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(
        outputs,
        (0..7).map(|v| json!(format!("ok-{v}"))).collect::<Vec<_>>()
    );
}

// --------------------------------------------------------------------- failing tools

/// Python: `test_parallel_tool_call_with_cancelled_sibling_reaches_final_output`. Rust has no
/// task cancellation inside a tool body; the closest analog is a tool whose future fails, which
/// the default error function turns into the generic message.
#[tokio::test]
async fn a_failing_sibling_tool_still_reaches_the_final_output() {
    let model = setup();
    model.add_turn(vec![
        function_call_with_id("ok_tool", "{}", "call_ok"),
        function_call_with_id("cancel_tool", "{}", "call_cancel"),
    ]);
    model.add_turn(vec![assistant_message("final answer")]);
    let cancel = FunctionTool::<()>::new("cancel_tool", "c", |_ctx, _a: AnyArgs| async {
        Err::<String, _>(AgentsError::other("tool-cancelled"))
    });
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(tool_returning::<()>("ok_tool", "ok"))
            .with_tool(cancel),
    );
    let result = Runner::run_with_options(agent, "user_message", RunOptions::new(()))
        .await
        .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("final answer"));
    assert_eq!(result.raw_responses.len(), 2);
    let outputs: Vec<Value> = input_of(model.calls().last().unwrap())
        .into_iter()
        .filter(|i| i["type"] == "function_call_output")
        .collect();
    assert_eq!(outputs.len(), 2);
    assert_eq!(outputs[0]["call_id"], "call_ok");
    assert_eq!(outputs[0]["output"], "ok");
    assert_eq!(outputs[1]["call_id"], "call_cancel");
    assert!(
        outputs[1]["output"]
            .as_str()
            .unwrap()
            .starts_with("An error occurred while running the tool. Please try again."),
        "{}",
        outputs[1]
    );
}

/// Python: `test_single_tool_call_with_cancelled_tool_reaches_final_output`.
#[tokio::test]
async fn a_single_failing_tool_still_reaches_the_final_output() {
    let model = setup();
    model.add_turn(vec![function_call_with_id(
        "cancel_tool",
        "{}",
        "call_cancel",
    )]);
    model.add_turn(vec![assistant_message("final answer")]);
    let cancel = FunctionTool::<()>::new("cancel_tool", "c", |_ctx, _a: AnyArgs| async {
        Err::<String, _>(AgentsError::other("tool-cancelled"))
    });
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(cancel),
    );
    let result = Runner::run_with_options(agent, "user_message", RunOptions::new(()))
        .await
        .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("final answer"));
    let outputs: Vec<Value> = input_of(model.calls().last().unwrap())
        .into_iter()
        .filter(|i| i["type"] == "function_call_output")
        .collect();
    assert_eq!(outputs.len(), 1);
    assert!(
        outputs[0]["output"]
            .as_str()
            .unwrap()
            .starts_with("An error occurred while running the tool. Please try again.")
    );
}

// --------------------------------------------------------------- reasoning id policy

fn omit_config() -> RunConfig<()> {
    let mut c = RunConfig::default();
    c.reasoning_item_id_policy = ReasoningItemIdPolicy::Omit;
    c
}

/// Python: `test_reasoning_item_id_policy_omits_follow_up_reasoning_ids`.
#[tokio::test]
async fn the_omit_policy_strips_follow_up_reasoning_ids() {
    let model = setup();
    model.add_turn(vec![
        reasoning("rs_first", Some("Thinking...")),
        function_call_with_id("foo", r#"{"a":"b"}"#, "call_first"),
    ]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(tool_returning::<()>("foo", "tool_result")),
    );
    let result = Runner::run_with_options(
        agent,
        "hello",
        RunOptions::new(()).run_config(omit_config()),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    let sent = find_reasoning(&input_of(model.calls().last().unwrap())).unwrap();
    assert!(sent.get("id").is_none(), "{sent}");
    let history: Vec<Value> = result.to_input_list().into_iter().map(|i| i.0).collect();
    assert!(find_reasoning(&history).unwrap().get("id").is_none());
}

/// Python: `test_call_model_input_filter_can_reintroduce_reasoning_ids`.
#[tokio::test]
async fn a_call_model_input_filter_can_reintroduce_reasoning_ids() {
    let model = setup();
    model.add_turn(vec![
        reasoning("rs_filter", Some("Thinking...")),
        function_call_with_id("foo", r#"{"a":"b"}"#, "call_filter"),
    ]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(tool_returning::<()>("foo", "tool_result")),
    );
    let mut config = omit_config();
    config.call_model_input_filter = Some(Arc::new(|data| {
        Box::pin(async move {
            let mut model_data = data.model_data;
            for item in &mut model_data.input {
                if item.item_type() == Some("reasoning") && item.0.get("id").is_none() {
                    item.0["id"] = json!("rs_reintroduced");
                }
            }
            Ok(model_data)
        })
    }));
    let result = Runner::run_with_options(agent, "hello", RunOptions::new(()).run_config(config))
        .await
        .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    let sent = find_reasoning(&input_of(model.calls().last().unwrap())).unwrap();
    assert_eq!(sent["id"], "rs_reintroduced");
    let history: Vec<Value> = result.to_input_list().into_iter().map(|i| i.0).collect();
    assert!(find_reasoning(&history).unwrap().get("id").is_none());
}

/// Python: `test_omit_policy_strips_reasoning_ids_already_stored_in_the_session`.
#[tokio::test]
async fn the_omit_policy_strips_reasoning_ids_already_in_the_session() {
    let model = setup();
    let specialist = Arc::new(Agent::<()>::new("specialist").with_model_instance(model.clone()));
    let triage = Arc::new(
        Agent::<()>::new("triage")
            .with_model_instance(model.clone())
            .with_handoff(handoff(specialist)),
    );
    let session = Arc::new(InMemorySession::new("issue-2020"));
    model.add_turn(vec![
        reasoning("rs_triage", None),
        handoff_call("specialist"),
    ]);
    model.add_turn(vec![assistant_message("handled")]);
    let first = Runner::run_with_options(
        triage.clone(),
        "hello",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    assert_eq!(first.final_output_text().as_deref(), Some("handled"));
    let stored: Vec<Value> = session
        .get_items(None)
        .await
        .unwrap()
        .into_iter()
        .map(|i| i.0)
        .collect();
    assert_eq!(find_reasoning(&stored).unwrap()["id"], "rs_triage");

    // The server no longer resolves the stored id: replaying it would fail.
    model.add_responder(|request| {
        if request.input.iter().any(|i| {
            i.item_type() == Some("reasoning") && i.0.get("id") == Some(&json!("rs_triage"))
        }) {
            return Err(AgentsError::other("Item with id 'rs_triage' not found."));
        }
        Ok(vec![assistant_message("done")])
    });
    let second = Runner::run_with_options(
        triage,
        "anything else?",
        RunOptions::new(())
            .session(session)
            .run_config(omit_config()),
    )
    .await
    .unwrap();
    assert_eq!(second.final_output_text().as_deref(), Some("done"));
    let replayed = find_reasoning(&input_of(model.calls().last().unwrap())).unwrap();
    assert!(replayed.get("id").is_none());
}

fn approval_tool(executed: Arc<AtomicUsize>) -> FunctionTool<()> {
    FunctionTool::new(
        "approval_tool",
        "needs approval",
        move |_ctx, _a: AnyArgs| {
            let executed = executed.clone();
            async move {
                executed.fetch_add(1, Ordering::SeqCst);
                Ok("ok".to_string())
            }
        },
    )
    .with_needs_approval(true)
}

/// Python: `test_resumed_run_uses_serialized_reasoning_item_id_policy`.
#[tokio::test]
async fn a_resumed_run_keeps_the_reasoning_id_policy() {
    let model = setup();
    model.add_turn(vec![
        reasoning("rs_resume", Some("Thinking...")),
        function_call_with_id("approval_tool", "{}", "call_resume"),
    ]);
    model.add_turn(vec![assistant_message("done")]);
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(approval_tool(Arc::new(AtomicUsize::new(0)))),
    );
    let first = Runner::run_with_options(
        agent.clone(),
        "hello",
        RunOptions::new(()).run_config(omit_config()),
    )
    .await
    .unwrap();
    assert_eq!(first.interruptions.len(), 1);
    let mut state = first.to_state();
    state.approve(&first.interruptions[0], false);
    let restored = RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();
    let resumed = Runner::run_with_options(agent, restored, RunOptions::new(()))
        .await
        .unwrap();
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    let sent = find_reasoning(&input_of(model.calls().last().unwrap())).unwrap();
    assert!(sent.get("id").is_none(), "{sent}");
}

// ------------------------------------------------------- pre-approval input guardrails

fn counting_guardrail(count: Arc<AtomicUsize>) -> ToolInputGuardrail<()> {
    ToolInputGuardrail::new("count", move |_d| {
        let count = count.clone();
        async move {
            count.fetch_add(1, Ordering::SeqCst);
            Ok(ToolGuardrailFunctionOutput::allow())
        }
    })
}

fn pre_approval_config() -> RunConfig<()> {
    let mut c = RunConfig::default();
    c.tool_execution = ToolExecutionConfig {
        pre_approval_tool_input_guardrails: true,
        ..Default::default()
    };
    c
}

/// Python: `test_pending_approval_skips_tool_input_guardrails_by_default`.
#[tokio::test]
async fn a_pending_approval_skips_tool_input_guardrails_by_default() {
    let model = setup();
    model.add_turn(vec![function_call_with_id(
        "approval_tool",
        "{}",
        "call_default",
    )]);
    let runs = Arc::new(AtomicUsize::new(0));
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(
                approval_tool(Arc::new(AtomicUsize::new(0)))
                    .with_input_guardrail(counting_guardrail(runs.clone())),
            ),
    );
    let result = Runner::run_with_options(agent, "hello", RunOptions::new(()))
        .await
        .unwrap();
    assert_eq!(result.interruptions.len(), 1);
    assert_eq!(runs.load(Ordering::SeqCst), 0);
    assert!(result.tool_input_guardrail_results.is_empty());
}

/// Python: `test_pre_approval_tool_input_guardrails_can_reject_before_pending_approval`.
#[tokio::test]
async fn pre_approval_guardrails_can_reject_before_the_approval() {
    let model = setup();
    model.add_turn(vec![function_call_with_id(
        "approval_tool",
        "{}",
        "call_reject",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let executed = Arc::new(AtomicUsize::new(0));
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(approval_tool(executed.clone()).with_input_guardrail(
                ToolInputGuardrail::new("reject", |_d| async {
                    Ok(ToolGuardrailFunctionOutput::reject_content(
                        "blocked before approval",
                    ))
                }),
            )),
    );
    let result = Runner::run_with_options(
        agent,
        "hello",
        RunOptions::new(()).run_config(pre_approval_config()),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("done"));
    assert!(result.interruptions.is_empty());
    assert_eq!(executed.load(Ordering::SeqCst), 0);
    assert_eq!(result.tool_input_guardrail_results.len(), 1);
    assert!(result.new_items.iter().any(|i| matches!(
        &i.kind,
        RunItemKind::ToolCallOutput { output, .. } if output == &json!("blocked before approval")
    )));
}

/// Python: `test_pre_approval_tool_input_guardrails_rerun_after_resume`.
#[tokio::test]
async fn pre_approval_guardrails_rerun_after_a_resume() {
    let model = setup();
    model.add_turn(vec![function_call_with_id(
        "approval_tool",
        "{}",
        "call_resume",
    )]);
    model.add_turn(vec![assistant_message("done")]);
    let runs = Arc::new(AtomicUsize::new(0));
    let executed = Arc::new(AtomicUsize::new(0));
    let agent = Arc::new(
        Agent::<()>::new("test")
            .with_model_instance(model.clone())
            .with_tool(
                approval_tool(executed.clone())
                    .with_input_guardrail(counting_guardrail(runs.clone())),
            ),
    );
    let first = Runner::run_with_options(
        agent.clone(),
        "hello",
        RunOptions::new(()).run_config(pre_approval_config()),
    )
    .await
    .unwrap();
    assert_eq!(first.interruptions.len(), 1);
    assert_eq!(runs.load(Ordering::SeqCst), 1);
    assert_eq!(executed.load(Ordering::SeqCst), 0);
    assert_eq!(first.tool_input_guardrail_results.len(), 1);

    let mut state = first.to_state();
    state.approve(&first.interruptions[0], false);
    let restored = RunState::from_json_str(&state.to_json_string().unwrap()).unwrap();
    let resumed = Runner::run_with_options(
        agent,
        restored,
        RunOptions::new(()).run_config(pre_approval_config()),
    )
    .await
    .unwrap();
    assert_eq!(resumed.final_output_text().as_deref(), Some("done"));
    assert_eq!(runs.load(Ordering::SeqCst), 2);
    assert_eq!(executed.load(Ordering::SeqCst), 1);
    assert_eq!(resumed.tool_input_guardrail_results.len(), 1);
}
