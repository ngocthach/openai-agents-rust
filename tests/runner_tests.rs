//! End-to-end runner behavior against a scripted model.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::agent::{AgentAsToolOptions, StopAtTools, ToolUseBehavior};
use openai_agents_sdk::items::RunItemKind;
use openai_agents_sdk::lifecycle::RunHooks;
use openai_agents_sdk::run_config::{ToolErrorKind, ToolNotFoundBehavior};
use openai_agents_sdk::run_error_handlers::{RunErrorHandlerResult, RunErrorHandlers};
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::testing::{
    FakeModel, assistant_message, function_call, function_call_with_id, refusal_message,
};
use openai_agents_sdk::tool::{FunctionTool, ToolTimeoutBehavior};
use openai_agents_sdk::tool_guardrails::{ToolGuardrailFunctionOutput, ToolInputGuardrail};
use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, InMemorySession, InputGuardrail, InputItem,
    OutputGuardrail, RunConfig, RunContext, RunOptions, RunState, Runner, Session, handoff,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

fn setup() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

#[derive(Deserialize, JsonSchema)]
struct CityArgs {
    city: String,
}

fn weather_tool<C: Send + Sync + 'static>() -> FunctionTool<C> {
    FunctionTool::new(
        "get_weather",
        "Get the weather",
        |_ctx, a: CityArgs| async move { Ok(format!("The weather in {} is sunny", a.city)) },
    )
}

#[tokio::test]
async fn plain_text_run() {
    let model = setup();
    model.add_turn(vec![assistant_message("Hello there")]);
    let agent = Agent::<()>::new("Assistant")
        .with_instructions("Be nice")
        .with_model_instance(model.clone());
    let result = Runner::run(&agent, "Hi").await.unwrap();
    assert_eq!(result.final_output_text().as_deref(), Some("Hello there"));
    assert_eq!(result.last_agent.name, "Assistant");
    assert_eq!(result.new_items.len(), 1);
    assert_eq!(result.usage().requests, 1);
    let call = model.last_call().unwrap();
    assert_eq!(call.system_instructions.as_deref(), Some("Be nice"));
    assert_eq!(call.input[0].text_content().as_deref(), Some("Hi"));
    assert_eq!(result.to_input_list().len(), 2);
    assert_eq!(result.last_response_id().as_deref(), Some("resp_fake_1"));
}

#[tokio::test]
async fn tool_loop_runs_tools_then_answers() {
    let model = setup();
    model.add_turn(vec![function_call("get_weather", r#"{"city":"Tokyo"}"#)]);
    model.add_turn(vec![assistant_message("It is sunny in Tokyo")]);
    let agent = Agent::<()>::new("Weather")
        .with_tool(weather_tool())
        .with_model_instance(model.clone());
    let result = Runner::run(&agent, "Weather in Tokyo?").await.unwrap();
    assert_eq!(
        result.final_output_text().as_deref(),
        Some("It is sunny in Tokyo")
    );
    let kinds: Vec<&str> = result.new_items.iter().map(|i| i.item_type()).collect();
    assert_eq!(
        kinds,
        vec![
            "tool_call_item",
            "tool_call_output_item",
            "message_output_item"
        ]
    );
    let second = &model.calls()[1];
    let last_input = second.input.last().unwrap();
    assert_eq!(last_input.item_type(), Some("function_call_output"));
    assert_eq!(last_input.0["output"], "The weather in Tokyo is sunny");
    assert_eq!(second.tools.len(), 1);
}

#[derive(Debug, Deserialize, JsonSchema, PartialEq)]
struct CalendarEvent {
    name: String,
    participants: Vec<String>,
}

#[tokio::test]
async fn structured_output_is_validated() {
    let model = setup();
    model.add_turn(vec![assistant_message(
        r#"{"name":"Party","participants":["A","B"]}"#,
    )]);
    let agent = Agent::<()>::new("Extractor")
        .with_output_type::<CalendarEvent>()
        .with_model_instance(model.clone());
    let result = Runner::run(&agent, "Party with A and B").await.unwrap();
    let ev: CalendarEvent = result.final_output_as().unwrap();
    assert_eq!(ev.participants, vec!["A", "B"]);
    assert!(model.last_call().unwrap().output_schema.is_some());

    model.add_turn(vec![assistant_message("not json")]);
    let err = Runner::run(&agent, "x").await.unwrap_err();
    assert!(matches!(err, AgentsError::ModelBehaviorError { .. }));
    // By default (like Python's `DONT_LOG_MODEL_DATA`) an invalid-JSON error is redacted and
    // carries no run data, so the model's payload never reaches the error.
    assert!(err.run_data().is_none());
    assert!(!err.to_string().contains("not json"));

    // With model-data logging enabled the same error names the payload and carries the run.
    openai_agents_sdk::debug::set_dont_log_model_data(false);
    model.add_turn(vec![assistant_message("not json")]);
    let err = Runner::run(&agent, "x").await.unwrap_err();
    openai_agents_sdk::debug::reset_dont_log_model_data();
    assert!(matches!(err, AgentsError::ModelBehaviorError { .. }));
    assert!(err.run_data().is_some());
}

#[tokio::test]
async fn invalid_output_and_refusal_handlers() {
    let model = setup();
    model.add_turn(vec![assistant_message("not json")]);
    model.add_turn(vec![refusal_message("I can't")]);
    let agent = Agent::<()>::new("Extractor")
        .with_output_type::<CalendarEvent>()
        .with_model_instance(model.clone());
    let handlers = RunErrorHandlers::new()
        .on_invalid_final_output(|_i| async {
            Ok(Some(RunErrorHandlerResult::new(
                json!({"name": "fallback", "participants": []}),
            )))
        })
        .on_model_refusal(|i| async move {
            let refusal = match &*i.error {
                AgentsError::ModelRefusalError { refusal, .. } => refusal.clone(),
                _ => String::new(),
            };
            Ok(Some(RunErrorHandlerResult::new(
                json!({"name": refusal, "participants": []}),
            )))
        });
    let r = Runner::run_with_options(
        &agent,
        "x",
        RunOptions::new(()).error_handlers(handlers.clone()),
    )
    .await
    .unwrap();
    assert_eq!(r.final_output.unwrap()["name"], "fallback");
    let r = Runner::run_with_options(&agent, "x", RunOptions::new(()).error_handlers(handlers))
        .await
        .unwrap();
    assert_eq!(r.final_output.unwrap()["name"], "I can't");
}

#[tokio::test]
async fn handoff_switches_agents() {
    let model = setup();
    let history = Arc::new(
        Agent::<()>::new("History Tutor")
            .with_handoff_description("History questions")
            .with_model_instance(model.clone()),
    );
    let triage = Agent::<()>::new("Triage")
        .with_handoff(handoff(history.clone()))
        .with_model_instance(model.clone());
    model.add_turn(vec![function_call("transfer_to_history_tutor", "{}")]);
    model.add_turn(vec![assistant_message("Washington")]);
    let result = Runner::run(&triage, "First president?").await.unwrap();
    assert_eq!(result.last_agent.name, "History Tutor");
    assert_eq!(result.final_output_text().as_deref(), Some("Washington"));
    let kinds: Vec<&str> = result.new_items.iter().map(|i| i.item_type()).collect();
    assert_eq!(
        kinds,
        vec![
            "handoff_call_item",
            "handoff_output_item",
            "message_output_item"
        ]
    );
    let first = &model.calls()[0];
    assert_eq!(first.handoffs[0].name, "transfer_to_history_tutor");
    let second = &model.calls()[1];
    let out = second.input.last().unwrap();
    assert_eq!(out.0["output"], r#"{"assistant":"History Tutor"}"#);
}

#[tokio::test]
async fn handoff_input_filter_and_nested_history() {
    let model = setup();
    let target = Arc::new(Agent::<()>::new("B").with_model_instance(model.clone()));
    let a =
        Agent::<()>::new("A")
            .with_tool(weather_tool())
            .with_handoff(handoff(target.clone()).with_input_filter(
                openai_agents_sdk::extensions::handoff_filters::remove_all_tools,
            ))
            .with_model_instance(model.clone());
    model.add_turn(vec![function_call("get_weather", r#"{"city":"X"}"#)]);
    model.add_turn(vec![function_call("transfer_to_b", "{}")]);
    model.add_turn(vec![assistant_message("done")]);
    Runner::run(&a, "hi").await.unwrap();
    let last = model.last_call().unwrap();
    assert!(
        last.input
            .iter()
            .all(|i| i.item_type() != Some("function_call"))
    );
    assert!(
        last.input
            .iter()
            .all(|i| i.item_type() != Some("function_call_output"))
    );

    // Nested history collapses the transcript into a summary message.
    let b = Arc::new(Agent::<()>::new("B").with_model_instance(model.clone()));
    let a = Agent::<()>::new("A")
        .with_handoff(handoff(b))
        .with_model_instance(model.clone());
    model.add_turn(vec![function_call("transfer_to_b", "{}")]);
    model.add_turn(vec![assistant_message("ok")]);
    let opts = RunOptions::new(()).run_config(RunConfig::new().with_nest_handoff_history(true));
    Runner::run_with_options(&a, "question", opts)
        .await
        .unwrap();
    let last = model.last_call().unwrap();
    assert_eq!(last.input.len(), 1);
    assert!(
        last.input[0]
            .text_content()
            .unwrap()
            .contains("<CONVERSATION HISTORY>")
    );
}

#[tokio::test]
async fn input_guardrail_tripwire_stops_run() {
    let model = setup();
    model.add_turn(vec![assistant_message("never")]);
    let g = InputGuardrail::new("math", |_c, _a, input: Vec<InputItem>| async move {
        let text = input[0].text_content().unwrap_or_default();
        Ok(GuardrailFunctionOutput::new(
            json!({"checked": true}),
            text.contains("2x"),
        ))
    })
    .blocking();
    let agent = Agent::<()>::new("Support")
        .with_input_guardrail(g)
        .with_model_instance(model.clone());
    let err = Runner::run(&agent, "solve 2x + 3 = 11").await.unwrap_err();
    match &err {
        AgentsError::InputGuardrailTripwireTriggered {
            guardrail_result, ..
        } => {
            assert_eq!(guardrail_result.guardrail_name, "math")
        }
        other => panic!("unexpected {other:?}"),
    }
    assert_eq!(
        model.calls().len(),
        0,
        "blocking guardrail must run before the model"
    );

    // Parallel guardrail that passes.
    let model2 = setup();
    model2.add_turn(vec![assistant_message("fine")]);
    let ok = InputGuardrail::new("ok", |_c, _a, _i| async {
        Ok(GuardrailFunctionOutput::pass())
    });
    let agent = Agent::<()>::new("S")
        .with_input_guardrail(ok)
        .with_model_instance(model2.clone());
    let r = Runner::run(&agent, "hello").await.unwrap();
    assert_eq!(r.input_guardrail_results.len(), 1);
}

#[tokio::test]
async fn output_guardrail_tripwire() {
    let model = setup();
    model.add_turn(vec![assistant_message("secret 42")]);
    let g = OutputGuardrail::new("no_numbers", |_c, _a, out: Value| async move {
        Ok(GuardrailFunctionOutput::new(
            Value::Null,
            out.as_str().unwrap_or("").contains("42"),
        ))
    });
    let agent = Agent::<()>::new("A")
        .with_output_guardrail(g)
        .with_model_instance(model.clone());
    let err = Runner::run(&agent, "x").await.unwrap_err();
    assert!(matches!(
        err,
        AgentsError::OutputGuardrailTripwireTriggered { .. }
    ));
}

#[tokio::test]
async fn max_turns_and_handler() {
    let model = setup();
    for _ in 0..3 {
        model.add_turn(vec![function_call("get_weather", r#"{"city":"X"}"#)]);
    }
    let agent = Agent::<()>::new("Loop")
        .with_tool(weather_tool())
        .with_model_instance(model.clone());
    let err = Runner::run_with_options(&agent, "x", RunOptions::new(()).max_turns(Some(2)))
        .await
        .unwrap_err();
    assert!(err.is_max_turns_exceeded());

    let model = setup();
    for _ in 0..2 {
        model.add_turn(vec![function_call("get_weather", r#"{"city":"X"}"#)]);
    }
    let agent = Agent::<()>::new("Loop")
        .with_tool(weather_tool())
        .with_model_instance(model.clone());
    let handlers = RunErrorHandlers::new().on_max_turns(|_i| async {
        Ok(Some(
            RunErrorHandlerResult::new("gave up").without_history(),
        ))
    });
    let r = Runner::run_with_options(
        &agent,
        "x",
        RunOptions::new(())
            .max_turns(Some(2))
            .error_handlers(handlers),
    )
    .await
    .unwrap();
    assert_eq!(r.final_output_text().as_deref(), Some("gave up"));
}

#[tokio::test]
async fn tool_use_behaviors() {
    let model = setup();
    model.add_turn(vec![function_call("get_weather", r#"{"city":"Paris"}"#)]);
    let agent = Agent::<()>::new("A")
        .with_tool(weather_tool())
        .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
        .with_model_instance(model.clone());
    let r = Runner::run(&agent, "x").await.unwrap();
    assert_eq!(
        r.final_output_text().as_deref(),
        Some("The weather in Paris is sunny")
    );

    model.add_turn(vec![function_call("get_weather", r#"{"city":"Rome"}"#)]);
    let agent = agent.clone_with(|a| {
        a.tool_use_behavior = ToolUseBehavior::StopAtTools(StopAtTools {
            stop_at_tool_names: vec!["get_weather".into()],
        })
    });
    let r = Runner::run(&agent, "x").await.unwrap();
    assert_eq!(
        r.final_output_text().as_deref(),
        Some("The weather in Rome is sunny")
    );
    assert_eq!(model.remaining_steps(), 0);
}

#[tokio::test]
async fn tool_errors_timeouts_and_not_found() {
    let model = setup();
    let failing = FunctionTool::<()>::new_no_args("boom", "fails", |_c| async {
        Err::<String, _>(AgentsError::other("kaboom"))
    });
    let slow = FunctionTool::<()>::new_no_args("slow", "slow", |_c| async {
        tokio::time::sleep(std::time::Duration::from_secs(5)).await;
        Ok("late")
    })
    .with_timeout(std::time::Duration::from_millis(20));
    model.add_turn(vec![
        function_call_with_id("boom", "{}", "c1"),
        function_call_with_id("slow", "{}", "c2"),
        function_call_with_id("missing", "{}", "c3"),
    ]);
    model.add_turn(vec![assistant_message("ok")]);
    let agent = Agent::<()>::new("A")
        .with_tools([failing, slow.clone()])
        .with_model_instance(model.clone());
    let opts = RunOptions::new(()).run_config(
        RunConfig::new()
            .with_tool_not_found_behavior(ToolNotFoundBehavior::ReturnErrorToModel)
            .with_tool_error_formatter(|args| {
                (args.kind == ToolErrorKind::ToolNotFound)
                    .then(|| format!("no tool {}", args.tool_name))
            }),
    );
    let r = Runner::run_with_options(&agent, "x", opts).await.unwrap();
    let outputs: Vec<Value> = r
        .new_items
        .iter()
        .filter_map(|i| i.tool_output().cloned())
        .collect();
    assert!(
        outputs
            .iter()
            .any(|o| o == "An error occurred while running the tool. Please try again.")
    );
    assert!(
        outputs
            .iter()
            .any(|o| o.as_str().unwrap_or("").contains("timed out"))
    );
    assert!(outputs.iter().any(|o| o == "no tool missing"));

    let model = setup();
    model.add_turn(vec![function_call("missing", "{}")]);
    let agent = Agent::<()>::new("A").with_model_instance(model.clone());
    assert!(matches!(
        Runner::run(&agent, "x").await.unwrap_err(),
        AgentsError::ModelBehaviorError { .. }
    ));

    let model = setup();
    model.add_turn(vec![function_call("slow", "{}")]);
    let agent = Agent::<()>::new("A")
        .with_tool(slow.with_timeout_behavior(ToolTimeoutBehavior::RaiseException))
        .with_model_instance(model.clone());
    assert!(matches!(
        Runner::run(&agent, "x").await.unwrap_err(),
        AgentsError::ToolTimeoutError { .. }
    ));
}

#[tokio::test]
async fn tool_input_guardrail_rejects_content() {
    let model = setup();
    let g = ToolInputGuardrail::new("no_secrets", |data| async move {
        if data.context.tool_arguments.contains("sk-") {
            Ok(ToolGuardrailFunctionOutput::reject_content(
                "Remove secrets",
            ))
        } else {
            Ok(ToolGuardrailFunctionOutput::allow())
        }
    });
    let tool = weather_tool::<()>().with_input_guardrail(g);
    model.add_turn(vec![function_call("get_weather", r#"{"city":"sk-123"}"#)]);
    model.add_turn(vec![assistant_message("ok")]);
    let agent = Agent::<()>::new("A")
        .with_tool(tool)
        .with_model_instance(model.clone());
    let r = Runner::run(&agent, "x").await.unwrap();
    assert_eq!(r.new_items[1].tool_output().unwrap(), "Remove secrets");
    assert_eq!(r.tool_input_guardrail_results.len(), 1);
}

#[tokio::test]
async fn approval_pause_and_resume_with_serialization() {
    let model = setup();
    let executed = Arc::new(AtomicUsize::new(0));
    let e2 = executed.clone();
    let tool = FunctionTool::<()>::new("cancel_order", "Cancel", move |_c, a: Value| {
        let e2 = e2.clone();
        async move {
            e2.fetch_add(1, Ordering::SeqCst);
            Ok(format!("cancelled {}", a["order_id"]))
        }
    })
    .with_params_json_schema(json!({"type":"object","properties":{"order_id":{"type":"integer"}},"required":["order_id"],"additionalProperties":false}))
    .with_needs_approval(true);
    let agent = Agent::<()>::new("Support")
        .with_tool(tool)
        .with_model_instance(model.clone());
    model.add_turn(vec![function_call_with_id(
        "cancel_order",
        r#"{"order_id":7}"#,
        "call_a",
    )]);
    let r = Runner::run(&agent, "cancel 7").await.unwrap();
    assert!(r.final_output.is_none());
    assert_eq!(r.interruptions.len(), 1);
    assert_eq!(r.interruptions[0].name(), "cancel_order");
    assert_eq!(executed.load(Ordering::SeqCst), 0);

    // Persist and reload the state, then approve.
    let json = r.to_state().to_json_string().unwrap();
    let mut state = RunState::from_json_str(&json).unwrap();
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);
    model.add_turn(vec![assistant_message("Order cancelled")]);
    let r2 = Runner::run(&agent, state).await.unwrap();
    assert_eq!(r2.final_output_text().as_deref(), Some("Order cancelled"));
    assert_eq!(executed.load(Ordering::SeqCst), 1);
    assert!(
        r2.new_items
            .iter()
            .any(|i| i.tool_output() == Some(&json!("cancelled 7")))
    );

    // Rejection with a custom message.
    model.add_turn(vec![function_call_with_id(
        "cancel_order",
        r#"{"order_id":8}"#,
        "call_b",
    )]);
    let r = Runner::run(&agent, "cancel 8").await.unwrap();
    let mut state = r.to_state();
    let item = state.get_interruptions()[0].clone();
    state.reject(&item, false, Some("Denied by reviewer".into()));
    model.add_turn(vec![assistant_message("Okay, not cancelled")]);
    let r2 = Runner::run(&agent, state).await.unwrap();
    assert!(
        r2.new_items
            .iter()
            .any(|i| i.tool_output() == Some(&json!("Denied by reviewer")))
    );
    assert_eq!(executed.load(Ordering::SeqCst), 1);
}

#[tokio::test]
async fn dynamic_approval_and_add_input() {
    let model = setup();
    let tool = weather_tool::<()>().with_needs_approval_fn(|_ctx, args, _id| async move {
        args["city"].as_str() == Some("Oakland")
    });
    let agent = Agent::<()>::new("W")
        .with_tool(tool)
        .with_model_instance(model.clone());
    model.add_turn(vec![function_call("get_weather", r#"{"city":"Oakland"}"#)]);
    let r = Runner::run(&agent, "weather").await.unwrap();
    assert_eq!(r.interruptions.len(), 1);
    let mut state = r.to_state();
    state.add_input("Also say hi").unwrap();
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, true);
    model.add_turn(vec![assistant_message("Sunny. Hi!")]);
    let r2 = Runner::run(&agent, state).await.unwrap();
    assert_eq!(r2.final_output_text().as_deref(), Some("Sunny. Hi!"));
    let last = model.last_call().unwrap();
    assert_eq!(
        last.input.last().unwrap().text_content().as_deref(),
        Some("Also say hi")
    );
    assert!(
        r2.new_items
            .iter()
            .any(|i| matches!(i.kind, RunItemKind::Input { .. }))
    );
}

#[tokio::test]
async fn agents_as_tools_and_nested_approvals() {
    let model = setup();
    let spanish = Agent::<()>::new("Spanish").with_model_instance(model.clone());
    let orchestrator = Agent::<()>::new("Orchestrator")
        .with_tool(spanish.as_tool("translate_to_spanish", "Translate to Spanish"))
        .with_model_instance(model.clone());
    model.add_turn(vec![function_call(
        "translate_to_spanish",
        r#"{"input":"hello"}"#,
    )]);
    model.add_turn(vec![assistant_message("hola")]);
    model.add_turn(vec![assistant_message("Spanish: hola")]);
    let r = Runner::run(&orchestrator, "Say hello in Spanish")
        .await
        .unwrap();
    assert_eq!(r.final_output_text().as_deref(), Some("Spanish: hola"));
    assert_eq!(
        model.calls()[1].input[0].text_content().as_deref(),
        Some("hello")
    );
    assert_eq!(r.usage().requests, 3);

    // A tool inside the nested agent needs approval; the interruption surfaces on the outer run.
    let model = setup();
    let inner = Agent::<()>::new("Inner")
        .with_tool(weather_tool().with_needs_approval(true))
        .with_model_instance(model.clone());
    let outer = Agent::<()>::new("Outer")
        .with_tool(inner.as_tool_with(AgentAsToolOptions::new().tool_name("inner_tool")))
        .with_model_instance(model.clone());
    model.add_turn(vec![function_call_with_id(
        "inner_tool",
        r#"{"input":"weather?"}"#,
        "outer_call",
    )]);
    model.add_turn(vec![function_call_with_id(
        "get_weather",
        r#"{"city":"Lima"}"#,
        "inner_call",
    )]);
    let r = Runner::run(&outer, "go").await.unwrap();
    assert_eq!(r.interruptions.len(), 1);
    assert_eq!(r.interruptions[0].agent, "Inner");
    let state_json = r.to_state().to_json().unwrap();
    let mut state = RunState::from_json(state_json).unwrap();
    let item = state.get_interruptions()[0].clone();
    state.approve(&item, false);
    model.add_turn(vec![assistant_message("Lima is sunny")]);
    model.add_turn(vec![assistant_message("Done: Lima is sunny")]);
    let r2 = Runner::run(&outer, state).await.unwrap();
    assert_eq!(
        r2.final_output_text().as_deref(),
        Some("Done: Lima is sunny")
    );
    let outer_output = r2.new_items.iter().find_map(|i| {
        i.tool_output()
            .filter(|o| o.as_str() == Some("Lima is sunny"))
    });
    assert!(outer_output.is_some());
}

#[tokio::test]
async fn sessions_persist_history() {
    let model = setup();
    let session: Arc<dyn Session> = Arc::new(InMemorySession::new("conv"));
    let agent = Agent::<()>::new("A").with_model_instance(model.clone());
    model.add_turn(vec![assistant_message("San Francisco")]);
    Runner::run_with_options(
        &agent,
        "City of Golden Gate?",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    model.add_turn(vec![assistant_message("California")]);
    let r = Runner::run_with_options(
        &agent,
        "Which state?",
        RunOptions::new(()).session(session.clone()),
    )
    .await
    .unwrap();
    assert_eq!(r.final_output_text().as_deref(), Some("California"));
    let last = model.last_call().unwrap();
    assert_eq!(last.input.len(), 3);
    assert_eq!(session.get_items(None).await.unwrap().len(), 4);
    let err = Runner::run_with_options(
        &agent,
        "x",
        RunOptions::new(())
            .session(session.clone())
            .conversation_id("conv_1"),
    )
    .await
    .unwrap_err();
    assert!(matches!(err, AgentsError::UserError { .. }));
}

#[tokio::test]
async fn previous_response_id_chaining_sends_only_new_items() {
    let model = setup();
    model.add_turn(vec![function_call("get_weather", r#"{"city":"Oslo"}"#)]);
    model.add_turn(vec![assistant_message("cold")]);
    let agent = Agent::<()>::new("A")
        .with_tool(weather_tool())
        .with_model_instance(model.clone());
    let r = Runner::run_with_options(
        &agent,
        "Oslo?",
        RunOptions::new(()).auto_previous_response_id(true),
    )
    .await
    .unwrap();
    assert_eq!(r.final_output_text().as_deref(), Some("cold"));
    let calls = model.calls();
    assert_eq!(calls[0].previous_response_id, None);
    assert_eq!(
        calls[1].previous_response_id.as_deref(),
        Some("resp_fake_1")
    );
    assert_eq!(calls[1].input.len(), 1);
    assert_eq!(calls[1].input[0].item_type(), Some("function_call_output"));
}

#[tokio::test]
async fn streaming_emits_events_in_order() {
    let model = setup();
    model.add_turn(vec![function_call("get_weather", r#"{"city":"Rome"}"#)]);
    model.add_turn(vec![assistant_message("Sunny in Rome")]);
    let agent = Agent::<()>::new("A")
        .with_tool(weather_tool())
        .with_model_instance(model.clone());
    let mut result = Runner::run_streamed(&agent, "Rome?");
    let mut names = Vec::new();
    let mut text = String::new();
    {
        let mut events = std::pin::pin!(result.stream_events());
        while let Some(ev) = events.next().await {
            let ev = ev.unwrap();
            if let Some(d) = ev.text_delta() {
                text.push_str(d);
            }
            match ev {
                StreamEvent::RunItem { name, .. } => names.push(name),
                StreamEvent::AgentUpdated { new_agent } => assert_eq!(new_agent.name, "A"),
                StreamEvent::RawResponse(_) => {}
            }
        }
    }
    assert!(result.is_complete());
    assert_eq!(text, "Sunny in Rome");
    assert_eq!(
        names,
        vec![
            RunItemStreamEventName::ToolCalled,
            RunItemStreamEventName::ToolOutput,
            RunItemStreamEventName::MessageOutputCreated
        ]
    );
    assert_eq!(result.final_output_text().as_deref(), Some("Sunny in Rome"));
}

#[tokio::test]
async fn streaming_surfaces_errors() {
    let model = setup();
    model.add_error("model exploded");
    let agent = Agent::<()>::new("A").with_model_instance(model.clone());
    let mut result = Runner::run_streamed(&agent, "x");
    let mut saw_err = false;
    {
        let mut events = std::pin::pin!(result.stream_events());
        while let Some(ev) = events.next().await {
            if ev.is_err() {
                saw_err = true;
            }
        }
    }
    assert!(saw_err);
    assert!(result.is_complete());
}

#[derive(Default)]
struct CountingHooks {
    events: Mutex<Vec<String>>,
}

#[async_trait::async_trait]
impl RunHooks<()> for CountingHooks {
    async fn on_agent_start(&self, _c: &RunContext<()>, a: &Agent<()>) {
        self.events
            .lock()
            .unwrap()
            .push(format!("start:{}", a.name));
    }
    async fn on_llm_start(
        &self,
        _c: &RunContext<()>,
        _a: &Agent<()>,
        _s: Option<&str>,
        _i: &[InputItem],
    ) {
        self.events.lock().unwrap().push("llm".into());
    }
    async fn on_tool_start(
        &self,
        c: &openai_agents_sdk::ToolContext<()>,
        _a: &Agent<()>,
        _t: &openai_agents_sdk::Tool<()>,
    ) {
        self.events
            .lock()
            .unwrap()
            .push(format!("tool:{}", c.tool_name));
    }
    async fn on_agent_end(&self, _c: &RunContext<()>, a: &Agent<()>, _o: &Value) {
        self.events.lock().unwrap().push(format!("end:{}", a.name));
    }
}

#[tokio::test]
async fn run_hooks_fire() {
    let model = setup();
    model.add_turn(vec![function_call("get_weather", r#"{"city":"Rome"}"#)]);
    model.add_turn(vec![assistant_message("ok")]);
    let agent = Agent::<()>::new("A")
        .with_tool(weather_tool())
        .with_model_instance(model.clone());
    let hooks = Arc::new(CountingHooks::default());
    Runner::run_with_options(&agent, "x", RunOptions::new(()).hooks(hooks.clone()))
        .await
        .unwrap();
    assert_eq!(
        *hooks.events.lock().unwrap(),
        vec!["start:A", "llm", "tool:get_weather", "llm", "end:A"]
    );
}

#[tokio::test]
async fn context_reaches_tools_and_reset_tool_choice() {
    struct UserInfo {
        name: String,
    }
    let model = setup();
    let tool = FunctionTool::<UserInfo>::new_no_args("whoami", "Name", |ctx| async move {
        Ok(ctx.context().name.clone())
    });
    let agent = Agent::<UserInfo>::new("A")
        .with_tool(tool)
        .with_model_settings(
            openai_agents_sdk::ModelSettings::default().with_tool_choice("required"),
        )
        .with_model_instance(model.clone());
    model.add_turn(vec![function_call("whoami", "{}")]);
    model.add_turn(vec![assistant_message("You are John")]);
    let r = Runner::run_with_options(
        &agent,
        "who?",
        RunOptions::new(UserInfo {
            name: "John".into(),
        }),
    )
    .await
    .unwrap();
    assert_eq!(r.new_items[1].tool_output().unwrap(), "John");
    let calls = model.calls();
    assert!(calls[0].model_settings.tool_choice.is_some());
    assert!(calls[1].model_settings.tool_choice.is_none());
}

#[tokio::test]
async fn call_model_input_filter_and_session_callback() {
    let model = setup();
    model.add_turn(vec![assistant_message("ok")]);
    let agent = Agent::<()>::new("A")
        .with_instructions("orig")
        .with_model_instance(model.clone());
    let cfg = RunConfig::new().with_call_model_input_filter(|data| async move {
        Ok(openai_agents_sdk::ModelInputData {
            input: data.model_data.input,
            instructions: Some("replaced".into()),
        })
    });
    Runner::run_with_options(&agent, "x", RunOptions::new(()).run_config(cfg))
        .await
        .unwrap();
    assert_eq!(
        model.last_call().unwrap().system_instructions.as_deref(),
        Some("replaced")
    );
}

#[tokio::test]
async fn model_retry_on_retryable_status() {
    let model = setup();
    model.enqueue(openai_agents_sdk::testing::ModelStep::ApiError(503));
    model.add_turn(vec![assistant_message("recovered")]);
    let agent = Agent::<()>::new("A")
        .with_model_settings(openai_agents_sdk::ModelSettings::default().with_retry(
            openai_agents_sdk::model_settings::ModelRetrySettings {
                max_retries: 2,
                backoff: openai_agents_sdk::model_settings::ModelRetryBackoffSettings {
                    initial_delay: 0.01,
                    max_delay: 0.01,
                    multiplier: 1.0,
                    jitter: false,
                },
                ..Default::default()
            },
        ))
        .with_model_instance(model.clone());
    let r = Runner::run(&agent, "x").await.unwrap();
    assert_eq!(r.final_output_text().as_deref(), Some("recovered"));
}

#[tokio::test]
async fn model_retry_policies_decide_and_respect_budget() {
    use openai_agents_sdk::model_settings::{ModelRetryBackoffSettings, ModelRetrySettings};
    use openai_agents_sdk::retry::{RetryDecision, RetryPolicy, retry_policies};
    use openai_agents_sdk::testing::ModelStep;
    let fast = ModelRetryBackoffSettings {
        initial_delay: 0.0,
        max_delay: 0.0,
        multiplier: 1.0,
        jitter: false,
    };
    let settings = |policy: RetryPolicy, max: u32| {
        openai_agents_sdk::ModelSettings::default().with_retry(
            ModelRetrySettings {
                max_retries: max,
                backoff: fast.clone(),
                ..Default::default()
            }
            .with_policy(policy),
        )
    };

    // A 418 is retried by an http_status policy.
    let model = setup();
    model.enqueue(ModelStep::ApiError(418));
    model.add_turn(vec![assistant_message("ok")]);
    let agent = Agent::<()>::new("A")
        .with_model_settings(settings(retry_policies::http_status([418]), 1))
        .with_model_instance(model.clone());
    assert_eq!(
        Runner::run(&agent, "x")
            .await
            .unwrap()
            .final_output_text()
            .as_deref(),
        Some("ok")
    );
    assert_eq!(model.calls().len(), 2);

    // The policy never retries a 503 and the budget limits a custom policy.
    let model = setup();
    model.enqueue(ModelStep::ApiError(503));
    let agent = Agent::<()>::new("A")
        .with_model_settings(settings(retry_policies::never(), 3))
        .with_model_instance(model.clone());
    assert!(Runner::run(&agent, "x").await.is_err());
    assert_eq!(model.calls().len(), 1);

    let model = setup();
    for _ in 0..3 {
        model.enqueue(ModelStep::ApiError(500));
    }
    let attempts = Arc::new(Mutex::new(Vec::new()));
    let a2 = attempts.clone();
    let custom = RetryPolicy::new(move |c| {
        a2.lock().unwrap().push(c.attempt);
        RetryDecision::retry().with_delay(0.0)
    });
    let agent = Agent::<()>::new("A")
        .with_model_settings(settings(custom, 2))
        .with_model_instance(model.clone());
    assert!(Runner::run(&agent, "x").await.is_err());
    assert_eq!(model.calls().len(), 3);
    assert_eq!(*attempts.lock().unwrap(), vec![1, 2]);
}

#[tokio::test]
async fn output_guardrail_redacts_blocked_terminal_tool_output() {
    use openai_agents_sdk::run_config::OutputGuardrailBlockedMessage;
    let model = setup();
    model.add_turn(vec![function_call("secret", "{}")]);
    let tool = FunctionTool::<()>::new_no_args("secret", "Secret", |_c| async { Ok("pin=1234") });
    let g = OutputGuardrail::new("no_pins", |_c, _a, out: Value| async move {
        Ok(GuardrailFunctionOutput::new(
            json!({"seen": out.clone()}),
            out.as_str().unwrap_or("").contains("pin="),
        ))
    });
    let agent = Agent::<()>::new("A")
        .with_tool(tool)
        .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
        .with_output_guardrail(g)
        .with_model_instance(model.clone());
    let session: Arc<dyn Session> = Arc::new(InMemorySession::new("s"));
    let config = RunConfig::new().with_output_guardrail_blocked_message(
        OutputGuardrailBlockedMessage::Text("[withheld]".into()),
    );
    let err = Runner::run_with_options(
        &agent,
        "show the secret",
        RunOptions::new(())
            .session(session.clone())
            .run_config(config),
    )
    .await
    .unwrap_err();
    let AgentsError::OutputGuardrailTripwireTriggered {
        guardrail_result, ..
    } = &err
    else {
        panic!("unexpected error {err}");
    };
    assert_eq!(guardrail_result.agent_output, json!("[withheld]"));
    assert!(guardrail_result.output.output_info.is_null());
    let stored: Vec<Value> = session
        .get_items(None)
        .await
        .unwrap()
        .into_iter()
        .map(|i| i.0)
        .collect();
    assert!(stored.iter().any(|i| i["type"] == "function_call"));
    let out = stored
        .iter()
        .find(|i| i["type"] == "function_call_output")
        .unwrap();
    assert_eq!(out["output"], "[withheld]");
    assert!(!serde_json::to_string(&stored).unwrap().contains("1234"));

    // A blocked assistant message is not persisted at all.
    model.add_turn(vec![assistant_message("pin=9999")]);
    let agent2 = Agent::<()>::new("B")
        .with_output_guardrail(OutputGuardrail::new(
            "no_pins",
            |_c, _a, out: Value| async move {
                Ok(GuardrailFunctionOutput::new(
                    Value::Null,
                    out.as_str().unwrap_or("").contains("pin="),
                ))
            },
        ))
        .with_model_instance(model.clone());
    let s2: Arc<dyn Session> = Arc::new(InMemorySession::new("s2"));
    assert!(
        Runner::run_with_options(&agent2, "hi", RunOptions::new(()).session(s2.clone()))
            .await
            .is_err()
    );
    let stored = serde_json::to_string(&s2.get_items(None).await.unwrap()).unwrap();
    assert!(stored.contains("hi"));
    assert!(!stored.contains("9999"));
}

#[tokio::test]
async fn tool_name_collisions_follow_policy() {
    use openai_agents_sdk::run_config::ToolNameCollisionPolicy;
    let model = setup();
    let other = Arc::new(Agent::<()>::new("B"));
    let dup = FunctionTool::<()>::new_no_args("transfer_to_b", "Dup", |_c| async { Ok("x") });
    let agent = Agent::<()>::new("A")
        .with_tool(dup)
        .with_handoff(handoff(other))
        .with_model_instance(model.clone());
    model.add_turn(vec![assistant_message("hi")]);
    Runner::run(&agent, "x").await.unwrap();
    let call = model.last_call().unwrap();
    assert!(call.tools.is_empty(), "the handoff wins the name");
    assert_eq!(call.handoffs.len(), 1);

    let config = RunConfig::new().with_tool_name_collision_policy(ToolNameCollisionPolicy::Error);
    let err = Runner::run_with_options(&agent, "x", RunOptions::new(()).run_config(config))
        .await
        .unwrap_err();
    assert!(
        err.to_string()
            .contains("Ambiguous tool routing configuration"),
        "{err}"
    );
}

#[test]
fn run_sync_works_outside_runtime() {
    let model = setup();
    model.add_turn(vec![assistant_message("sync ok")]);
    let agent = Agent::<()>::new("A").with_model_instance(model.clone());
    let r = Runner::run_sync(&agent, "x").unwrap();
    assert_eq!(r.final_output_text().as_deref(), Some("sync ok"));
}
