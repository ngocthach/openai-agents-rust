//! Ports of the construction, enablement, output and input cases of `test_agent_as_tool.py`.
//!
//! Python monkeypatches `Runner.run` with dummy results; these tests drive a real nested run
//! with a scripted model instead, and call `FunctionTool::invoke` where Python calls
//! `on_invoke_tool` directly. Truthiness tests (`__bool__` returning `False`), builders that
//! return a wrong type, and async input builders cannot exist in Rust: callbacks are `Option`s
//! and typed.

use std::sync::{Arc, Mutex, OnceLock};

use async_trait::async_trait;
use openai_agents_sdk::agent::{AgentAsToolOptions, StructuredToolInput};
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::lifecycle::RunHooks;
use openai_agents_sdk::memory::{InMemorySession, Session};
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::{RunConfig, ToolNameCollisionPolicy};
use openai_agents_sdk::testing::{
    FakeModel, assistant_message, function_call, function_call_with_id,
};
use openai_agents_sdk::tool::{FunctionTool, IsEnabled, tool_namespace};
use openai_agents_sdk::{Agent, AgentsError, RunContext, Runner, ToolContext};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

#[derive(Clone, Default)]
struct BoolCtx {
    enable_tools: bool,
}

fn enabled_when_ctx_allows() -> IsEnabled<BoolCtx> {
    IsEnabled::Dynamic(Arc::new(|ctx: RunContext<BoolCtx>, _agent: String| {
        Box::pin(async move { ctx.context().enable_tools })
    }))
}

fn tool_names<C>(tools: &[openai_agents_sdk::tool::Tool<C>]) -> Vec<String> {
    tools.iter().map(|t| t.name()).collect()
}

fn direct_ctx(tool_name: &str, call_id: &str, args: &str) -> ToolContext<()> {
    let mut ctx = ToolContext::direct(RunContext::new(()), "outer", tool_name, args);
    ctx.tool_call_id = call_id.to_string();
    ctx
}

async fn invoke_text(tool: &FunctionTool<()>, ctx: ToolContext<()>, args: &str) -> String {
    match tool.invoke(ctx, args.to_string()).await.unwrap() {
        openai_agents_sdk::tool::ToolOutput::Text(text) => text,
        other => panic!("expected text output, got {other:?}"),
    }
}

// ---------------------------------------------------------------- name collisions

fn refund_tool(agent_name: &str, description: &str) -> FunctionTool<()> {
    Agent::<()>::new(agent_name)
        .as_tool_with(AgentAsToolOptions::new().tool_description(description.to_string()))
}

async fn run_with_policy(
    tools: Vec<FunctionTool<()>>,
    policy: ToolNameCollisionPolicy,
) -> (Arc<FakeModel>, Result<(), AgentsError>) {
    let model = setup();
    model.add_turn(vec![assistant_message("done")]);
    let orchestrator = Agent::<()>::new("orchestrator")
        .with_tools(tools)
        .with_model_instance(model.clone());
    let config = RunConfig::<()>::default().with_tool_name_collision_policy(policy);
    let result =
        Runner::run_with_options(&orchestrator, "go", RunOptions::new(()).run_config(config))
            .await
            .map(|_| ());
    (model, result)
}

/// Python: `test_agent_as_tool_rejects_colliding_derived_names`.
#[tokio::test]
async fn colliding_derived_names_are_rejected_with_an_exact_message() {
    let (model, result) = run_with_policy(
        vec![
            refund_tool("Refund", "First refund agent"),
            refund_tool("refund", "Second refund agent"),
        ],
        ToolNameCollisionPolicy::Error,
    )
    .await;
    let error = result.expect_err("the collision is an error");
    assert!(matches!(error, AgentsError::UserError { .. }), "{error:?}");
    assert_eq!(
        error.to_string(),
        "Ambiguous agent tool configuration: agents 'Refund' and 'refund' both derive the tool \
         name `refund`. Pass an explicit `tool_name=` to one of them."
    );
    assert!(model.calls().is_empty());
}

/// Python: `test_agent_as_tool_derived_name_collision_allows_explicit_override`.
#[tokio::test]
async fn an_explicit_tool_name_resolves_a_derived_name_collision() {
    let orchestrator = Arc::new(
        Agent::<()>::new("orchestrator")
            .with_tool(refund_tool("Refund", "First refund agent"))
            .with_tool(
                Agent::<()>::new("refund").as_tool_with(
                    AgentAsToolOptions::new()
                        .tool_name("normalized_refund")
                        .tool_description("Second refund agent"),
                ),
            ),
    );
    let tools = orchestrator
        .get_all_tools(&RunContext::new(()))
        .await
        .unwrap();
    assert_eq!(tool_names(&tools), ["refund", "normalized_refund"]);
}

/// Python: `test_agent_as_tool_rejects_distinct_agents_with_the_same_name`.
#[tokio::test]
async fn distinct_agents_with_the_same_name_are_rejected() {
    let (_, result) = run_with_policy(
        vec![
            refund_tool("Refund", "First refund agent"),
            refund_tool("Refund", "Second refund agent"),
        ],
        ToolNameCollisionPolicy::Error,
    )
    .await;
    let error = result.expect_err("the collision is an error");
    assert!(
        error
            .to_string()
            .contains("the tool name `refund` is used by multiple tools"),
        "{error}"
    );
}

static RECORDS: Mutex<Vec<String>> = Mutex::new(Vec::new());
static LOG_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

struct Capture;

impl log::Log for Capture {
    fn enabled(&self, _: &log::Metadata<'_>) -> bool {
        true
    }
    fn log(&self, record: &log::Record<'_>) {
        if record.level() == log::Level::Warn && record.target() == "openai.agents" {
            RECORDS.lock().unwrap().push(format!("{}", record.args()));
        }
    }
    fn flush(&self) {}
}

fn install_logger() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let _ = log::set_logger(&Capture);
        log::set_max_level(log::LevelFilter::Trace);
    });
}

/// Python: `test_agent_as_tool_warns_and_keeps_last_distinct_agent_with_the_same_name`.
#[tokio::test]
async fn a_warning_keeps_the_last_of_two_distinct_agents_with_the_same_name() {
    let _lock = LOG_LOCK.lock().await;
    install_logger();
    openai_agents_sdk::debug::set_dont_log_tool_data(false);
    RECORDS.lock().unwrap().clear();
    let (model, result) = run_with_policy(
        vec![
            refund_tool("Refund", "First refund agent"),
            refund_tool("Refund", "Second refund agent"),
        ],
        ToolNameCollisionPolicy::Warn,
    )
    .await;
    openai_agents_sdk::debug::reset_dont_log_tool_data();
    result.unwrap();

    let sent = model.calls()[0].tools.clone();
    assert_eq!(sent.len(), 1);
    match &sent[0] {
        openai_agents_sdk::models::interface::ToolSpec::Function(f) => {
            assert_eq!(f.name, "refund");
            assert_eq!(f.description, "Second refund agent");
        }
        other => panic!("unexpected tool {other:?}"),
    }
    let ambiguous: Vec<String> = RECORDS
        .lock()
        .unwrap()
        .iter()
        .filter(|m| m.starts_with("Ambiguous function tool configuration"))
        .cloned()
        .collect();
    assert_eq!(
        ambiguous,
        [
            "Ambiguous function tool configuration: the tool name `refund` is used by multiple \
             tools. Assign a unique routed name to every colliding function tool with \
             `name_override=`, `tool_name=`, or a namespace."
        ]
    );
}

/// Python: `test_agent_as_tool_ignores_disabled_derived_name_collision`.
#[tokio::test]
async fn a_disabled_tool_does_not_collide() {
    let orchestrator = Arc::new(
        Agent::<()>::new("orchestrator")
            .with_tool(refund_tool("Refund", "First refund agent"))
            .with_tool(
                Agent::<()>::new("refund").as_tool_with(
                    AgentAsToolOptions::new()
                        .tool_description("Second refund agent")
                        .is_enabled(false),
                ),
            ),
    );
    let tools = orchestrator
        .get_all_tools(&RunContext::new(()))
        .await
        .unwrap();
    assert_eq!(tool_names(&tools), ["refund"]);
}

/// Python: `test_agent_as_tool_default_identity_tracks_the_current_tool_name`.
#[tokio::test]
async fn a_copied_default_name_collides_but_a_renamed_tool_does_not() {
    let derived = refund_tool("Refund", "Refund agent");
    let copied = derived.clone();
    let mut renamed = derived.clone();
    renamed.name = "renamed_refund".to_string();
    let colliding = || refund_tool("refund", "Normalized refund agent");

    let (_, result) =
        run_with_policy(vec![copied, colliding()], ToolNameCollisionPolicy::Error).await;
    let error = result.expect_err("a copied derived name still collides");
    assert!(
        error
            .to_string()
            .contains("Ambiguous agent tool configuration"),
        "{error}"
    );

    let (_, result) =
        run_with_policy(vec![renamed, colliding()], ToolNameCollisionPolicy::Error).await;
    result.expect("a renamed tool no longer collides");
}

/// Python: `test_agent_as_tool_derived_names_are_disambiguated_by_namespace`.
#[tokio::test]
async fn a_namespace_disambiguates_derived_names() {
    let sales = tool_namespace(
        "sales",
        "Sales",
        vec![refund_tool("Refund", "Sales refunds")],
    )
    .unwrap();
    let support = tool_namespace(
        "support",
        "Support",
        vec![refund_tool("refund", "Support refunds")],
    )
    .unwrap();
    let orchestrator = Arc::new(
        Agent::<()>::new("orchestrator")
            .with_tools(sales)
            .with_tools(support),
    );
    let tools = orchestrator
        .get_all_tools(&RunContext::new(()))
        .await
        .unwrap();
    let names: Vec<String> = tools
        .iter()
        .map(|t| t.as_function().expect("a function tool").qualified_name())
        .collect();
    assert_eq!(names, ["sales.refund", "support.refund"]);
}

// ------------------------------------------------------------------- enablement

/// Python: `test_agent_as_tool_is_enabled_mixed`.
#[tokio::test]
async fn enablement_can_mix_static_and_dynamic_checks() {
    let agent = Agent::<BoolCtx>::new("test_agent");
    let as_tool = |name: &str, enabled: Option<IsEnabled<BoolCtx>>| {
        let mut options = AgentAsToolOptions::new()
            .tool_name(name.to_string())
            .tool_description(format!("{name} tool"));
        if let Some(enabled) = enabled {
            options = options.is_enabled(enabled);
        }
        agent.as_tool_with(options)
    };
    let orchestrator = Arc::new(
        Agent::<BoolCtx>::new("orchestrator")
            .with_tool(as_tool("always_enabled", Some(IsEnabled::Static(true))))
            .with_tool(as_tool("always_disabled", Some(IsEnabled::Static(false))))
            .with_tool(as_tool(
                "conditionally_enabled",
                Some(enabled_when_ctx_allows()),
            ))
            .with_tool(as_tool("default_enabled", None)),
    );

    let off = orchestrator
        .get_all_tools(&RunContext::new(BoolCtx {
            enable_tools: false,
        }))
        .await
        .unwrap();
    assert_eq!(tool_names(&off), ["always_enabled", "default_enabled"]);

    let on = orchestrator
        .get_all_tools(&RunContext::new(BoolCtx { enable_tools: true }))
        .await
        .unwrap();
    assert_eq!(
        tool_names(&on),
        ["always_enabled", "conditionally_enabled", "default_enabled"]
    );
}

/// Python: `test_agent_as_tool_is_enabled_preserves_other_params`.
#[tokio::test]
async fn enablement_does_not_disturb_the_other_options() {
    let agent = Agent::<BoolCtx>::new("test_agent");
    let tool = agent.as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("custom_tool_name")
            .tool_description("A custom tool with all parameters")
            .custom_output_extractor(|result| async move {
                Ok(format!(
                    "CUSTOM: {}",
                    result.final_output_text().unwrap_or_default()
                ))
            })
            .is_enabled(true),
    );
    assert_eq!(tool.name, "custom_tool_name");
    assert_eq!(tool.description, "A custom tool with all parameters");
    assert!(matches!(tool.is_enabled, IsEnabled::Static(true)));

    let orchestrator = Arc::new(Agent::<BoolCtx>::new("orchestrator").with_tool(tool));
    let tools = orchestrator
        .get_all_tools(&RunContext::new(BoolCtx { enable_tools: true }))
        .await
        .unwrap();
    assert_eq!(tool_names(&tools), ["custom_tool_name"]);
}

// ---------------------------------------------------------------- output and runs

fn nested_agent(name: &str, model: &Arc<FakeModel>) -> Agent<()> {
    Agent::<()>::new(name).with_model_instance(model.clone())
}

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

#[derive(Debug, Default)]
struct RecordingHooks {
    agents_started: Mutex<Vec<String>>,
}

#[async_trait]
impl RunHooks<()> for RecordingHooks {
    async fn on_agent_start(&self, _ctx: &RunContext<()>, agent: &Agent<()>) {
        self.agents_started.lock().unwrap().push(agent.name.clone());
    }
}

/// Python: `test_agent_as_tool_custom_output_extractor` (the options reach the nested run).
#[tokio::test]
async fn run_options_reach_the_nested_run() {
    let model = setup();
    model.add_turn(vec![assistant_message("Original text")]);
    let summarizer = nested_agent("summarizer", &model);
    let hooks = Arc::new(RecordingHooks::default());
    let session = Arc::new(InMemorySession::new("sess_123"));
    let extracted_for = Arc::new(Mutex::new(Vec::<(String, String, String)>::new()));
    let record = extracted_for.clone();
    let tool = summarizer.as_tool_with({
        let mut options = AgentAsToolOptions::new()
            .tool_name("summary_tool")
            .tool_description("Summarize input")
            .max_turns(7)
            .session(session.clone())
            .custom_output_extractor(move |result| {
                let record = record.clone();
                async move {
                    let invocation = result.agent_tool_invocation.clone().expect("an invocation");
                    record.lock().unwrap().push((
                        invocation.tool_call_id,
                        invocation.tool_name,
                        invocation.tool_arguments,
                    ));
                    Ok("custom output".to_string())
                }
            });
        options.hooks = Some(hooks.clone());
        options
    });

    let args = r#"{"input": "summarize this"}"#;
    let output = invoke_text(&tool, direct_ctx("summary_tool", "call_2", args), args).await;
    assert_eq!(output, "custom output", "{output}");

    assert_eq!(*hooks.agents_started.lock().unwrap(), ["summarizer"]);
    let call = &model.calls()[0];
    assert_eq!(
        call.input[0].text_content().as_deref(),
        Some("summarize this")
    );
    assert!(!session.get_items(None).await.unwrap().is_empty());
    assert_eq!(
        *extracted_for.lock().unwrap(),
        [(
            "call_2".to_string(),
            "summary_tool".to_string(),
            args.to_string()
        )]
    );
}

/// Python: `test_agent_as_tool_custom_output_extractor` (`previous_response_id`,
/// `conversation_id` and `max_turns`).
#[tokio::test]
async fn the_response_ids_and_turn_limit_reach_the_nested_run() {
    let args = r#"{"input": "hi"}"#;
    let model = setup();
    model.add_turn(vec![assistant_message("ok")]);
    let mut options = AgentAsToolOptions::new()
        .tool_name("conv_tool")
        .max_turns(7);
    options.conversation_id = Some("conv_1".to_string());
    let tool = nested_agent("conv_agent", &model).as_tool_with(options);
    invoke_text(&tool, direct_ctx("conv_tool", "call_conv", args), args).await;
    assert_eq!(model.calls()[0].conversation_id.as_deref(), Some("conv_1"));

    let model = setup();
    model.add_turn(vec![assistant_message("ok")]);
    let mut options = AgentAsToolOptions::new().tool_name("prev_tool");
    options.previous_response_id = Some("resp_1".to_string());
    let tool = nested_agent("prev_agent", &model).as_tool_with(options);
    invoke_text(&tool, direct_ctx("prev_tool", "call_prev", args), args).await;
    assert_eq!(
        model.calls()[0].previous_response_id.as_deref(),
        Some("resp_1")
    );

    // The turn limit bounds the nested run.
    let model = setup();
    for _ in 0..8 {
        model.add_turn(vec![function_call("again", "{}")]);
    }
    let looping = nested_agent("looping", &model).with_tool(FunctionTool::new(
        "again",
        "loops",
        |_ctx, _a: NoArgs| async { Ok("again".to_string()) },
    ));
    let tool = looping.as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("loop_tool")
            .max_turns(7)
            .failure_error_function(|_ctx, error| format!("failed: {error:?}")),
    );
    let output = invoke_text(&tool, direct_ctx("loop_tool", "call_loop", args), args).await;
    assert!(output.contains("MaxTurnsExceeded"), "{output}");
    assert_eq!(model.calls().len(), 7);
}

/// Python: `test_agent_as_tool_fallback_uses_current_run_items_only`.
#[tokio::test]
async fn an_empty_final_output_falls_back_to_the_latest_nonempty_message() {
    let model = setup();
    model.add_turn(vec![function_call_with_id("inner", "{}", "call_current")]);
    model.add_turn(vec![
        assistant_message("Current run summary"),
        assistant_message(""),
    ]);
    let agent = nested_agent("summarizer", &model).with_tool(FunctionTool::new(
        "inner",
        "inner",
        |_ctx, _a: NoArgs| async { Ok("Current tool output".to_string()) },
    ));
    let tool = agent.as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("summary_tool")
            .tool_description("Summarize current run output"),
    );
    let args = r#"{"input": "hello"}"#;
    let output = invoke_text(&tool, direct_ctx("summary_tool", "call_1", args), args).await;
    assert_eq!(output, "Current run summary");
}

/// Python: `test_agent_as_tool_fallback_returns_most_recent_current_run_output`.
#[tokio::test]
async fn an_empty_final_output_falls_back_to_the_most_recent_item() {
    let model = setup();
    model.add_turn(vec![
        assistant_message("Older message output"),
        function_call_with_id("inner", "{}", "call_current"),
    ]);
    model.add_turn(vec![assistant_message("")]);
    let agent = nested_agent("summarizer", &model).with_tool(FunctionTool::new(
        "inner",
        "inner",
        |_ctx, _a: NoArgs| async { Ok("Newest tool output".to_string()) },
    ));
    let tool = agent.as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("summary_tool")
            .tool_description("Summarize current run output"),
    );
    let args = r#"{"input": "hello"}"#;
    let output = invoke_text(&tool, direct_ctx("summary_tool", "call_1", args), args).await;
    assert_eq!(output, "Newest tool output");
}

/// Python: `test_agent_as_tool_extractor_can_access_agent_tool_invocation`.
#[tokio::test]
async fn the_extractor_sees_the_agent_tool_invocation() {
    let model = setup();
    model.add_turn(vec![assistant_message("done")]);
    let seen = Arc::new(Mutex::new(None));
    let record = seen.clone();
    let tool = nested_agent("nested_agent", &model).as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("nested_tool")
            .tool_description("A nested agent tool")
            .custom_output_extractor(move |result| {
                let record = record.clone();
                async move {
                    *record.lock().unwrap() = result.agent_tool_invocation.clone();
                    Ok("extracted".to_string())
                }
            }),
    );
    let args = r#"{"input": "hello"}"#;
    let output = invoke_text(&tool, direct_ctx("nested_tool", "call_abc_123", args), args).await;
    assert_eq!(output, "extracted");
    let invocation = seen.lock().unwrap().clone().expect("an invocation");
    assert_eq!(invocation.tool_call_id, "call_abc_123");
    assert_eq!(invocation.tool_name, "nested_tool");
    assert_eq!(invocation.tool_arguments, args);
}

// ------------------------------------------------------------ run config inheritance

fn ctx_with_config(
    tool_name: &str,
    call_id: &str,
    args: &str,
    config: RunConfig<()>,
) -> ToolContext<()> {
    direct_ctx(tool_name, call_id, args).with_run_config(Arc::new(config))
}

/// Python: `test_agent_as_tool_inherits_parent_run_config_when_not_set` and
/// `test_agent_as_tool_inherits_trace_include_sensitive_data_setting`.
#[tokio::test]
async fn the_nested_run_inherits_the_parent_run_config() {
    use openai_agents_sdk::models::ModelTracing;
    let model = setup();
    model.add_turn(vec![assistant_message("ok")]);
    let tool = nested_agent("inherits_config_agent", &model).as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("inherits_config_tool")
            .tool_description("inherit config"),
    );
    let args = r#"{"input":"hello"}"#;
    let parent = RunConfig::<()>::default()
        .with_trace_include_sensitive_data(false)
        .with_tracing_disabled(false);
    let output = invoke_text(
        &tool,
        ctx_with_config("inherits_config_tool", "call_inherit", args, parent),
        args,
    )
    .await;
    assert_eq!(output, "ok");
    assert_eq!(model.calls()[0].tracing, ModelTracing::EnabledWithoutData);
}

/// Python: `test_agent_as_tool_explicit_run_config_overrides_parent_context`.
#[tokio::test]
async fn an_explicit_run_config_overrides_the_parent_run_config() {
    use openai_agents_sdk::models::ModelTracing;
    let model = setup();
    model.add_turn(vec![assistant_message("ok")]);
    let explicit = RunConfig::<()>::default()
        .with_trace_include_sensitive_data(true)
        .with_tracing_disabled(false);
    let tool = nested_agent("override_config_agent", &model).as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("override_config_tool")
            .tool_description("override config")
            .run_config(explicit),
    );
    let args = r#"{"input":"hello"}"#;
    let parent = RunConfig::<()>::default()
        .with_trace_include_sensitive_data(false)
        .with_tracing_disabled(false);
    let output = invoke_text(
        &tool,
        ctx_with_config("override_config_tool", "call_override", args, parent),
        args,
    )
    .await;
    assert_eq!(output, "ok");
    assert_eq!(model.calls()[0].tracing, ModelTracing::Enabled);
}

// -------------------------------------------------------------- structured input

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct TranslationInput {
    text: String,
    source: String,
    target: String,
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct DescribedInput {
    /// Text to translate
    text: String,
    /// Target language
    target: String,
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct TextOnly {
    text: String,
}

/// Python: `test_agent_as_tool_structured_input_sets_tool_input`.
#[tokio::test]
async fn structured_input_is_passed_as_json_and_captured_as_tool_input() {
    let model = setup();
    model.add_turn(vec![function_call_with_id("peek", "{}", "peek_call")]);
    model.add_turn(vec![assistant_message("ok")]);
    let nested_tool_input = Arc::new(Mutex::new(None::<Value>));
    let record = nested_tool_input.clone();
    let agent = nested_agent("translator", &model).with_tool(FunctionTool::new(
        "peek",
        "peek",
        move |ctx, _a: NoArgs| {
            let record = record.clone();
            async move {
                *record.lock().unwrap() = ctx.run_context().tool_input().cloned();
                Ok("peeked".to_string())
            }
        },
    ));
    let tool = agent.as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("translate")
            .tool_description("Translate text")
            .parameters::<TranslationInput>(),
    );

    let args = json!({"text": "hola", "source": "es", "target": "en"});
    let parent = RunContext::new(());
    let ctx = ToolContext::direct(parent.clone(), "outer", "translate", &args.to_string());
    tool.invoke(ctx, args.to_string()).await.unwrap();

    let sent = model.calls()[0].input[0]
        .text_content()
        .expect("text input");
    let parsed: Value =
        serde_json::from_str(sent.split("```").nth(1).expect("a fenced block").trim()).unwrap();
    assert_eq!(parsed, args);
    assert_eq!(nested_tool_input.lock().unwrap().clone(), Some(args));
    assert!(parent.tool_input().is_none());
}

/// Python: `test_agent_as_tool_clears_stale_tool_input_for_plain_tools`.
#[tokio::test]
async fn a_plain_agent_tool_clears_a_stale_tool_input() {
    let model = setup();
    model.add_turn(vec![function_call_with_id("peek", "{}", "peek_call")]);
    model.add_turn(vec![assistant_message("ok")]);
    let nested_tool_input = Arc::new(Mutex::new(Some(json!("unset"))));
    let record = nested_tool_input.clone();
    let agent = nested_agent("plain_agent", &model).with_tool(FunctionTool::new(
        "peek",
        "peek",
        move |ctx, _a: NoArgs| {
            let record = record.clone();
            async move {
                *record.lock().unwrap() = ctx.run_context().tool_input().cloned();
                Ok("peeked".to_string())
            }
        },
    ));
    let tool = agent.as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("plain_tool")
            .tool_description("Plain tool"),
    );

    let stale = RunContext::new(()).with_tool_input(json!({"text": "bonjour"}));
    let args = r#"{"input": "hello"}"#;
    tool.invoke(
        ToolContext::direct(stale.clone(), "outer", "plain_tool", args),
        args.to_string(),
    )
    .await
    .unwrap();

    assert_eq!(nested_tool_input.lock().unwrap().clone(), None);
    assert_eq!(stale.tool_input(), Some(&json!({"text": "bonjour"})));
}

/// Python: `test_agent_as_tool_includes_schema_summary_with_descriptions`.
#[tokio::test]
async fn the_schema_summary_lists_described_fields() {
    let model = setup();
    model.add_turn(vec![assistant_message("ok")]);
    let tool = nested_agent("summary_agent", &model).as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("summarize_schema")
            .tool_description("Summary tool")
            .parameters::<DescribedInput>(),
    );
    let args = json!({"text": "hola", "target": "en"}).to_string();
    tool.invoke(
        direct_ctx("summarize_schema", "call_summary", &args),
        args.clone(),
    )
    .await
    .unwrap();

    let sent = model.calls()[0].input[0].text_content().unwrap();
    assert!(sent.contains("Input Schema Summary:"), "{sent}");
    assert!(sent.contains("text (string, required)"), "{sent}");
    assert!(sent.contains("Text to translate"), "{sent}");
    assert!(sent.contains("target (string, required)"), "{sent}");
    assert!(sent.contains("Target language"), "{sent}");
    assert!(sent.contains(r#""text": "hola""#), "{sent}");
    assert!(sent.contains(r#""target": "en""#), "{sent}");
}

/// Python: `test_agent_as_tool_supports_custom_input_builder` (the builder sees the params, no
/// summary and no schema; a builder may return input items).
#[tokio::test]
async fn a_custom_input_builder_can_supply_input_items() {
    let model = setup();
    model.add_turn(vec![assistant_message("ok")]);
    let calls = Arc::new(Mutex::new(
        Vec::<(Value, Option<String>, Option<Value>)>::new(),
    ));
    let record = calls.clone();
    let custom_items = vec![InputItem(
        json!({"role": "user", "content": "custom input"}),
    )];
    let returned = custom_items.clone();
    let tool = nested_agent("builder_agent", &model).as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("builder_tool")
            .tool_description("Builder tool")
            .parameters::<TextOnly>()
            .input_items_builder(move |input: &StructuredToolInput<'_>| {
                record.lock().unwrap().push((
                    input.params.clone(),
                    input.summary.map(str::to_string),
                    input.json_schema.cloned(),
                ));
                returned.clone()
            }),
    );
    let args = json!({"text": "hola"});
    tool.invoke(
        direct_ctx("builder_tool", "call_builder", &args.to_string()),
        args.to_string(),
    )
    .await
    .unwrap();

    assert_eq!(model.calls()[0].input, custom_items);
    let calls = calls.lock().unwrap();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0], (args, None, None));
}

/// Python: `test_agent_as_tool_includes_json_schema_when_requested`.
#[tokio::test]
async fn include_input_schema_embeds_the_json_schema() {
    let model = setup();
    model.add_turn(vec![assistant_message("ok")]);
    let tool = nested_agent("schema_agent", &model).as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("schema_tool")
            .tool_description("Schema tool")
            .parameters::<DescribedInput>()
            .include_input_schema(true),
    );
    let args = json!({"text": "hola", "target": "en"}).to_string();
    tool.invoke(
        direct_ctx("schema_tool", "call_schema", &args),
        args.clone(),
    )
    .await
    .unwrap();
    let sent = model.calls()[0].input[0].text_content().unwrap();
    assert!(sent.contains("Input JSON Schema:"), "{sent}");
    assert!(sent.contains(r#""properties""#), "{sent}");
    assert!(sent.contains(r#""text""#), "{sent}");
    assert!(sent.contains(r#""target""#), "{sent}");
}

/// Python: `test_agent_as_tool_ignores_input_schema_without_parameters`.
#[tokio::test]
async fn include_input_schema_is_ignored_without_parameters() {
    let model = setup();
    model.add_turn(vec![assistant_message("ok")]);
    let tool = nested_agent("default_schema_agent", &model).as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("default_schema_tool")
            .tool_description("Default schema tool")
            .include_input_schema(true),
    );
    let args = r#"{"input": "hello"}"#;
    tool.invoke(
        direct_ctx("default_schema_tool", "call_default_schema", args),
        args.to_string(),
    )
    .await
    .unwrap();
    assert_eq!(
        model.calls()[0].input[0].text_content().as_deref(),
        Some("hello")
    );
    assert!(tool.params_json_schema.get("properties").is_some());
}

/// Python: `test_replaced_agent_as_tool_normal_failure_uses_replaced_policy`.
#[tokio::test]
async fn a_copied_agent_tool_uses_the_failure_policy_it_was_given() {
    let model = setup();
    model.add_error("test failure");
    let tool = nested_agent("failing_agent", &model).as_tool_with(
        AgentAsToolOptions::new()
            .tool_name("failing_agent_tool")
            .tool_description("Agent tool that raises"),
    );
    let replaced = tool.clone().without_failure_error_function();
    let args = r#"{"input": "hello"}"#;
    let error = replaced
        .invoke(direct_ctx(&replaced.name, "call_1", args), args.to_string())
        .await
        .expect_err("the failure is raised");
    assert!(error.to_string().contains("test failure"), "{error}");
}

/// Python: `test_replaced_agent_as_tool_invalid_input_uses_replaced_name` (and the schema check
/// the default `{"input": str}` parameters apply to every call).
#[tokio::test]
async fn invalid_agent_tool_input_names_the_replaced_tool() {
    let model = setup();
    let mut replaced = nested_agent("nested_agent", &model)
        .as_tool_with(
            AgentAsToolOptions::new()
                .tool_name("nested_agent_tool")
                .tool_description("Nested agent tool"),
        )
        .without_failure_error_function();
    replaced.name = "replaced_nested_agent_tool".to_string();

    for args in ["{}", r#"{"input": 3}"#, r#"["input"]"#] {
        let error = replaced
            .invoke(direct_ctx(&replaced.name, "call_1", args), args.to_string())
            .await
            .expect_err("the arguments do not match the schema");
        assert!(
            matches!(error, AgentsError::ModelBehaviorError { .. }),
            "{error:?}"
        );
        assert!(
            error
                .to_string()
                .contains("Invalid JSON input for tool replaced_nested_agent_tool"),
            "{args}: {error}"
        );
    }
    assert!(model.calls().is_empty());
}
