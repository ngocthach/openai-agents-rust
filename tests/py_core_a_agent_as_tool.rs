//! Ports of the behavioral half of `test_agent_as_tool.py`.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::agent::{AgentAsToolOptions, AgentToolStreamEvent, StructuredToolInput};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call};
use openai_agents_sdk::tool::{FunctionTool, IsEnabled};
use openai_agents_sdk::{Agent, RunContext, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::json;

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

#[derive(Clone, Default)]
struct BoolCtx {
    enable_tools: bool,
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct Empty {}

fn enabled_when_ctx_allows() -> IsEnabled<BoolCtx> {
    IsEnabled::Dynamic(Arc::new(|ctx: RunContext<BoolCtx>, _agent: String| {
        Box::pin(async move { ctx.context().enable_tools })
    }))
}

#[tokio::test]
async fn agent_as_tool_default_name_is_derived_from_the_agent_name() {
    let agent = Agent::<()>::new("My Agent");
    let tool = agent.as_tool_with(AgentAsToolOptions::new());
    assert_eq!(tool.name, "my_agent");
}

#[tokio::test]
async fn agent_as_tool_is_enabled_bool() {
    let agent = Agent::<BoolCtx>::new("test_agent");
    let orchestrator = Arc::new(
        Agent::<BoolCtx>::new("orchestrator")
            .with_tool(
                agent.as_tool_with(
                    AgentAsToolOptions::new()
                        .tool_name("disabled_agent_tool")
                        .tool_description("A disabled agent tool")
                        .is_enabled(false),
                ),
            )
            .with_tool(
                agent.as_tool_with(
                    AgentAsToolOptions::new()
                        .tool_name("enabled_agent_tool")
                        .tool_description("An enabled agent tool")
                        .is_enabled(true),
                ),
            )
            .with_tool(
                agent.as_tool_with(
                    AgentAsToolOptions::new()
                        .tool_name("default_agent_tool")
                        .tool_description("A default agent tool"),
                ),
            ),
    );

    let ctx = RunContext::new(BoolCtx { enable_tools: true });
    let tools = orchestrator.get_all_tools(&ctx).await.unwrap();
    let names: Vec<String> = tools.iter().map(|t| t.name()).collect();
    assert!(names.contains(&"enabled_agent_tool".to_string()));
    assert!(names.contains(&"default_agent_tool".to_string()));
    assert!(!names.contains(&"disabled_agent_tool".to_string()));
}

#[tokio::test]
async fn agent_as_tool_is_enabled_callable() {
    let agent = Agent::<BoolCtx>::new("test_agent");
    let orchestrator = Arc::new(
        Agent::<BoolCtx>::new("orchestrator")
            .with_tool(
                agent.as_tool_with(
                    AgentAsToolOptions::new()
                        .tool_name("conditional_agent_tool")
                        .tool_description("conditional")
                        .is_enabled(enabled_when_ctx_allows()),
                ),
            )
            .with_tool(
                agent.as_tool_with(
                    AgentAsToolOptions::new()
                        .tool_name("lambda_agent_tool")
                        .tool_description("lambda")
                        .is_enabled(enabled_when_ctx_allows()),
                ),
            ),
    );

    let disabled = RunContext::new(BoolCtx {
        enable_tools: false,
    });
    assert!(
        orchestrator
            .get_all_tools(&disabled)
            .await
            .unwrap()
            .is_empty()
    );

    let enabled = RunContext::new(BoolCtx { enable_tools: true });
    let tools = orchestrator.get_all_tools(&enabled).await.unwrap();
    let names: Vec<String> = tools.iter().map(|t| t.name()).collect();
    assert_eq!(names.len(), 2);
    assert!(names.contains(&"conditional_agent_tool".to_string()));
    assert!(names.contains(&"lambda_agent_tool".to_string()));
}

#[tokio::test]
async fn agent_as_tool_returns_final_output() {
    let model = setup();
    model.add_turn(vec![function_call("story_tool", r#"{"input":"hello"}"#)]);
    model.add_turn(vec![assistant_message("Hello world")]);
    model.add_turn(vec![assistant_message("outer done")]);

    let storyteller = Agent::<()>::new("storyteller").with_model_instance(model.clone());
    let orchestrator = Agent::<()>::new("orchestrator")
        .with_tool(
            storyteller.as_tool_with(
                AgentAsToolOptions::new()
                    .tool_name("story_tool")
                    .tool_description("Tell a short story"),
            ),
        )
        .with_model_instance(model.clone());

    let result = Runner::run(&orchestrator, "go").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("outer done"));
    let nested_output = result
        .new_items
        .iter()
        .find_map(|i| i.tool_output().cloned())
        .expect("nested tool output");
    assert_eq!(nested_output, json!("Hello world"));
    // The nested run received the structured `input` field as its prompt.
    assert_eq!(
        model.calls()[1].input[0].text_content().as_deref(),
        Some("hello")
    );
}

#[tokio::test]
async fn agent_as_tool_custom_output_extractor() {
    let model = setup();
    model.add_turn(vec![function_call("summary_tool", r#"{"input":"hello"}"#)]);
    model.add_turn(vec![assistant_message("Original text")]);
    model.add_turn(vec![assistant_message("outer done")]);

    let summarizer = Agent::<()>::new("summarizer").with_model_instance(model.clone());
    let orchestrator = Agent::<()>::new("orchestrator")
        .with_tool(
            summarizer.as_tool_with(
                AgentAsToolOptions::new()
                    .tool_name("summary_tool")
                    .tool_description("Summarize")
                    .custom_output_extractor(|result| async move {
                        Ok(format!(
                            "Extracted: {}",
                            result.final_output_text().unwrap_or_default()
                        ))
                    }),
            ),
        )
        .with_model_instance(model.clone());

    let result = Runner::run(&orchestrator, "go").await.unwrap();
    let nested_output = result
        .new_items
        .iter()
        .find_map(|i| i.tool_output().cloned())
        .expect("nested tool output");
    assert_eq!(nested_output, json!("Extracted: Original text"));
}

#[tokio::test]
async fn agent_as_tool_structured_input_sets_tool_input() {
    #[derive(Deserialize, JsonSchema)]
    #[allow(dead_code)]
    struct Params {
        topic: String,
        length: u32,
    }

    let model = setup();
    model.add_turn(vec![function_call(
        "structured_tool",
        r#"{"topic":"otters","length":2}"#,
    )]);
    model.add_turn(vec![assistant_message("nested done")]);
    model.add_turn(vec![assistant_message("outer done")]);

    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let recorder = seen.clone();
    let inner_tool = FunctionTool::new("inner", "inner", move |ctx, _a: Empty| {
        let recorder = recorder.clone();
        async move {
            recorder.lock().unwrap().push(ctx.tool_arguments.clone());
            Ok("ok".to_string())
        }
    });

    let inner = Agent::<()>::new("inner_agent")
        .with_tool(inner_tool)
        .with_model_instance(model.clone());
    let orchestrator = Agent::<()>::new("orchestrator")
        .with_tool(
            inner.as_tool_with(
                AgentAsToolOptions::new()
                    .tool_name("structured_tool")
                    .tool_description("structured")
                    .parameters::<Params>(),
            ),
        )
        .with_model_instance(model.clone());

    let result = Runner::run(&orchestrator, "go").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("outer done"));

    // The declared schema reaches the model.
    let first_call = &model.calls()[0];
    let spec = first_call
        .tools
        .iter()
        .find_map(|t| match t {
            openai_agents_sdk::models::interface::ToolSpec::Function(f)
                if f.name == "structured_tool" =>
            {
                Some(f.clone())
            }
            _ => None,
        })
        .expect("structured_tool is advertised");
    assert_eq!(
        spec.parameters["properties"]["topic"]["type"],
        json!("string")
    );
    assert!(
        spec.parameters["required"]
            .as_array()
            .unwrap()
            .contains(&json!("topic"))
    );
}

#[derive(Deserialize, JsonSchema)]
#[allow(dead_code)]
struct BuilderParams {
    input: String,
}

#[tokio::test]
async fn agent_as_tool_input_builder_alone_opts_into_structured_input() {
    let model = setup();
    model.add_turn(vec![function_call("solo_tool", r#"{"input":"hello"}"#)]);
    model.add_turn(vec![assistant_message("nested done")]);
    model.add_turn(vec![assistant_message("outer done")]);
    let inner = Agent::<()>::new("inner_agent").with_model_instance(model.clone());
    let orchestrator = Agent::<()>::new("orchestrator")
        .with_tool(
            inner.as_tool_with(
                AgentAsToolOptions::new()
                    .tool_name("solo_tool")
                    .input_builder(|_input: &StructuredToolInput<'_>| "built input".to_string()),
            ),
        )
        .with_model_instance(model.clone());
    Runner::run(&orchestrator, "go").await.unwrap();
    assert_eq!(
        model.calls()[1].input[0].text_content().as_deref(),
        Some("built input")
    );
}

#[tokio::test]
async fn agent_as_tool_supports_custom_input_builder() {
    let model = setup();
    model.add_turn(vec![function_call("built_tool", r#"{"input":"hello"}"#)]);
    model.add_turn(vec![assistant_message("nested done")]);
    model.add_turn(vec![assistant_message("outer done")]);

    let inner = Agent::<()>::new("inner_agent").with_model_instance(model.clone());
    let orchestrator = Agent::<()>::new("orchestrator")
        .with_tool(
            inner.as_tool_with(
                AgentAsToolOptions::new()
                    .tool_name("built_tool")
                    .tool_description("built")
                    .parameters::<BuilderParams>()
                    .input_builder(|input: &StructuredToolInput<'_>| {
                        let topic = input
                            .params
                            .get("input")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default();
                        format!("Please handle: {topic}")
                    }),
            ),
        )
        .with_model_instance(model.clone());

    Runner::run(&orchestrator, "go").await.unwrap();
    assert_eq!(
        model.calls()[1].input[0].text_content().as_deref(),
        Some("Please handle: hello")
    );
}

#[tokio::test]
async fn agent_as_tool_failure_error_function_custom_handler() {
    let model = setup();
    model.add_turn(vec![function_call("failing_tool", r#"{"input":"hello"}"#)]);
    model.add_error("nested boom");
    model.add_turn(vec![assistant_message("outer done")]);

    let inner = Agent::<()>::new("inner_agent").with_model_instance(model.clone());
    let orchestrator = Agent::<()>::new("orchestrator")
        .with_tool(
            inner.as_tool_with(
                AgentAsToolOptions::new()
                    .tool_name("failing_tool")
                    .tool_description("fails")
                    .failure_error_function(|_ctx: &RunContext<()>, error| {
                        format!("handled: {error}")
                    }),
            ),
        )
        .with_model_instance(model.clone());

    let result = Runner::run(&orchestrator, "go").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("outer done"));
    let nested_output = result
        .new_items
        .iter()
        .find_map(|i| i.tool_output().cloned())
        .expect("nested tool output");
    assert!(
        nested_output
            .as_str()
            .unwrap_or_default()
            .starts_with("handled:"),
        "unexpected output: {nested_output}"
    );
}

#[tokio::test]
async fn agent_as_tool_failure_error_function_none_reraises() {
    let model = setup();
    model.add_turn(vec![function_call("failing_tool", r#"{"input":"hello"}"#)]);
    model.add_error("nested boom");

    let inner = Agent::<()>::new("inner_agent").with_model_instance(model.clone());
    let orchestrator = Agent::<()>::new("orchestrator")
        .with_tool(
            inner
                .as_tool_with(
                    AgentAsToolOptions::new()
                        .tool_name("failing_tool")
                        .tool_description("fails"),
                )
                .without_failure_error_function(),
        )
        .with_model_instance(model.clone());

    let err = Runner::run(&orchestrator, "go").await.unwrap_err();
    assert!(err.to_string().contains("nested boom"), "{err}");
}

#[tokio::test]
async fn agent_as_tool_streams_events_with_on_stream() {
    let model = setup();
    model.add_turn(vec![function_call("stream_tool", r#"{"input":"hello"}"#)]);
    model.add_turn(vec![assistant_message("nested done")]);
    model.add_turn(vec![assistant_message("outer done")]);

    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let recorder = seen.clone();
    let inner = Agent::<()>::new("inner_agent").with_model_instance(model.clone());
    let orchestrator = Agent::<()>::new("orchestrator")
        .with_tool(
            inner.as_tool_with(
                AgentAsToolOptions::new()
                    .tool_name("stream_tool")
                    .tool_description("streams")
                    .on_stream(move |event: AgentToolStreamEvent<()>| {
                        let recorder = recorder.clone();
                        async move {
                            recorder
                                .lock()
                                .unwrap()
                                .push(event.event.event_type().to_string());
                        }
                    }),
            ),
        )
        .with_model_instance(model.clone());

    let result = Runner::run(&orchestrator, "go").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("outer done"));
    let events = seen.lock().unwrap().clone();
    assert!(
        events.iter().any(|e| e == "run_item_stream_event"),
        "nested run item events reach the handler: {events:?}"
    );
}
