//! Ports found by the completeness sweep of the top-level Python test files:
//! `test_exception_exports.py::test_run_error_details_str_uses_pretty_printer` and
//! `test_agent_as_tool_callback_validation.py::test_agent_as_tool_preserves_callbacks_and_keyword_run_config`.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::agent::{AgentAsToolOptions, AgentToolStreamEvent};
use openai_agents_sdk::guardrail::{GuardrailFunctionOutput, InputGuardrail};
use openai_agents_sdk::items::InputItem;
use openai_agents_sdk::run_config::{ModelInputData, RunConfig};
use openai_agents_sdk::run_context::RunContext;
use openai_agents_sdk::testing::{FakeModel, assistant_message};
use openai_agents_sdk::tool_context::ToolContext;
use openai_agents_sdk::{Agent, RunErrorDetails};

#[test]
fn run_error_details_display_uses_the_pretty_printer() {
    let details = RunErrorDetails {
        last_agent: "worker".into(),
        ..Default::default()
    };
    let expected = "RunErrorDetails:\n- Last agent: Agent(name=\"worker\", ...)\n- 0 new item(s)\n- 0 raw response(s)\n- 0 input guardrail result(s)\n- 0 output guardrail result(s)\n- 0 tool input guardrail result(s)\n- 0 tool output guardrail result(s)\n(See `RunErrorDetails` for more details)";
    assert_eq!(details.to_string(), expected);
    assert_eq!(details.pretty_print(), expected);
}

async fn nested_run(with_handler: bool) {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    let events: Arc<Mutex<Vec<String>>> = Arc::default();
    let filtered: Arc<Mutex<Vec<String>>> = Arc::default();
    let checked: Arc<Mutex<Vec<String>>> = Arc::default();

    let checked_sink = checked.clone();
    let check_input = InputGuardrail::new("check_input", move |_c, agent: Arc<Agent<()>>, _i| {
        let checked_sink = checked_sink.clone();
        async move {
            checked_sink.lock().unwrap().push(agent.name.clone());
            Ok(GuardrailFunctionOutput::new(serde_json::Value::Null, false))
        }
    })
    .blocking();
    let filtered_sink = filtered.clone();
    let config = RunConfig::<()>::new()
        .with_input_guardrail(check_input)
        .with_call_model_input_filter(move |data| {
            let filtered_sink = filtered_sink.clone();
            async move {
                filtered_sink.lock().unwrap().push(data.agent.name.clone());
                Ok(ModelInputData {
                    input: vec![InputItem::user("filtered")],
                    instructions: None,
                })
            }
        })
        .with_tracing_disabled(true);

    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![assistant_message("done")]);
    let agent = Agent::<()>::new("worker").with_model_instance(model.clone());
    let mut options = AgentAsToolOptions::new()
        .tool_name("worker")
        .tool_description("Run the worker")
        .run_config(config);
    if with_handler {
        let sink = events.clone();
        options = options.on_stream(move |event: AgentToolStreamEvent<()>| {
            let sink = sink.clone();
            async move {
                sink.lock()
                    .unwrap()
                    .push(event.event.event_type().to_string());
            }
        });
    }
    let nested_tool = agent.as_tool_with(options);

    let args = r#"{"input":"original"}"#;
    let context = ToolContext::direct(RunContext::new(()), "outer", "worker", args);
    let output = nested_tool.invoke(context, args.to_string()).await.unwrap();

    assert_eq!(output.to_model_string(), "done");
    assert_eq!(*checked.lock().unwrap(), vec!["worker".to_string()]);
    assert_eq!(*filtered.lock().unwrap(), vec!["worker".to_string()]);
    let calls = model.calls();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].input, vec![InputItem::user("filtered")]);
    assert!(calls[0].tracing.is_disabled());
    assert_eq!(!events.lock().unwrap().is_empty(), with_handler);
}

#[tokio::test]
async fn agent_as_tool_preserves_the_run_config_without_a_stream_handler() {
    nested_run(false).await;
}

#[tokio::test]
async fn agent_as_tool_preserves_the_run_config_and_the_stream_handler() {
    nested_run(true).await;
}
