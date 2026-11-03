//! Ports of the SDK-level cases of `tests/test_example_workflows.py`. Those tests compose the
//! example scripts' patterns (judge loop, deterministic flow, routing, guardrails, forced tool
//! use, agents-as-tools) from public SDK pieces and a scripted model; they do not need the example
//! modules, so they run through the Rust API with `ScriptedModel`.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use openai_agents_sdk::agent::{ToolUseBehavior, ToolsToFinalOutputResult};
use openai_agents_sdk::handoffs::handoff;
use openai_agents_sdk::items::OutputItem;
use openai_agents_sdk::model_settings::{ModelSettings, ToolChoice};
use openai_agents_sdk::result::CancelMode;
use openai_agents_sdk::run_context::RunContext;
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_full,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, AgentsError, GuardrailFunctionOutput, InputGuardrail, InputItem, OutputGuardrail,
    RunOptions, Runner,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

fn text(text: &str) -> ScriptedStep {
    ScriptedStep::output(vec![assistant_message(text)])
}

fn calls(calls: Vec<OutputItem>) -> ScriptedStep {
    ScriptedStep::output(calls)
}

fn tool_call(name: &str, args: Value, call_id: &str) -> OutputItem {
    function_call_full(name, args.to_string(), call_id, None, None)
}

fn model(steps: Vec<ScriptedStep>) -> Arc<ScriptedModel> {
    let model = Arc::new(ScriptedModel::new());
    model.extend(steps).unwrap();
    model
}

fn agent(name: &str, model: &Arc<ScriptedModel>) -> Agent<()> {
    Agent::<()>::new(name).with_model_instance(model.clone())
}

fn user_input(content: &str) -> Vec<InputItem> {
    vec![InputItem::user(content)]
}

#[derive(Deserialize, JsonSchema)]
struct EvaluationFeedback {
    feedback: String,
    score: String,
}

#[derive(Deserialize, JsonSchema)]
struct OutlineCheckerOutput {
    good_quality: bool,
    is_scifi: bool,
}

/// Python: `test_llm_as_judge_loop_handles_dataclass_feedback`.
#[tokio::test]
async fn the_judge_loop_runs_until_the_evaluator_passes() {
    let outline_model = model(vec![text("Outline v1"), text("Outline v2")]);
    let judge_model = model(vec![
        text(&json!({"feedback": "Add more suspense", "score": "needs_improvement"}).to_string()),
        text(&json!({"feedback": "Looks good", "score": "pass"}).to_string()),
    ]);
    let outline_agent = agent("outline", &outline_model);
    let judge_agent = agent("judge", &judge_model).with_output_type::<EvaluationFeedback>();

    let mut conversation = user_input("Tell me a space story");
    let mut latest_outline = String::new();
    for (expected_outline, expected_score) in
        [("Outline v1", "needs_improvement"), ("Outline v2", "pass")]
    {
        let outline = Runner::run(&outline_agent, conversation.clone())
            .await
            .unwrap();
        latest_outline = outline.final_output_text().unwrap();
        assert_eq!(latest_outline, expected_outline);
        conversation = outline.to_input_list();

        let judged = Runner::run(&judge_agent, conversation.clone())
            .await
            .unwrap();
        let feedback: EvaluationFeedback = judged.final_output_as().unwrap();
        assert_eq!(feedback.score, expected_score);
        if feedback.score == "pass" {
            break;
        }
        conversation.push(InputItem::user(format!("Feedback: {}", feedback.feedback)));
    }
    assert_eq!(latest_outline, "Outline v2");
    assert_eq!(conversation.len(), 4);
    assert_eq!(judge_model.last_call().unwrap().request.input, conversation);
}

/// Python: `test_parallel_translation_flow_reuses_runner_outputs`.
#[tokio::test]
async fn translations_feed_a_picker_agent() {
    let translation_model = model(vec![text("Uno"), text("Dos"), text("Tres")]);
    let spanish = agent("spanish_agent", &translation_model);
    let picker_model = model(vec![text("Pick: Dos")]);
    let picker = agent("picker", &picker_model);

    let mut translations = Vec::new();
    for _ in 0..3 {
        let result = Runner::run(&spanish, "Hello").await.unwrap();
        translations.push(result.final_output_text().unwrap());
    }
    let combined = translations.join("\n\n");
    let prompt = format!("Input: Hello\n\nTranslations:\n{combined}");
    let picked = Runner::run(&picker, prompt.as_str()).await.unwrap();

    assert_eq!(translations, ["Uno", "Dos", "Tres"]);
    assert_eq!(picked.final_output_text().unwrap(), "Pick: Dos");
    assert_eq!(
        picker_model.last_call().unwrap().request.input,
        vec![InputItem::user(prompt)]
    );
}

/// Python: `test_deterministic_story_flow_stops_when_checker_blocks`.
#[tokio::test]
async fn a_failed_quality_gate_never_runs_the_story_agent() {
    let outline_model = model(vec![text("Outline v1")]);
    let checker_model = model(vec![text(
        &json!({"good_quality": false, "is_scifi": true}).to_string(),
    )]);
    let story_model = model(vec![ScriptedStep::raise_error(AgentsError::user(
        "story should not run",
    ))]);
    let outline = agent("outline", &outline_model);
    let checker = agent("checker", &checker_model).with_output_type::<OutlineCheckerOutput>();
    let story = agent("story", &story_model);

    let outline_result = Runner::run(&outline, user_input("Sci-fi please"))
        .await
        .unwrap();
    let checked = Runner::run(&checker, outline_result.to_input_list())
        .await
        .unwrap();
    let decision: OutlineCheckerOutput = checked.final_output_as().unwrap();
    assert!(!decision.good_quality);
    assert!(decision.is_scifi);
    if decision.good_quality && decision.is_scifi {
        Runner::run(&story, outline_result.final_output_text().unwrap().as_str())
            .await
            .unwrap();
    }
    assert!(
        story_model.calls().is_empty(),
        "story agent must stay gated"
    );
}

/// Python: `test_deterministic_story_flow_runs_story_on_pass`.
#[tokio::test]
async fn a_passed_quality_gate_runs_the_story_agent() {
    let outline_model = model(vec![text("Outline ready")]);
    let checker_model = model(vec![text(
        &json!({"good_quality": true, "is_scifi": true}).to_string(),
    )]);
    let story_model = model(vec![text("Final story")]);
    let outline = agent("outline", &outline_model);
    let checker = agent("checker", &checker_model).with_output_type::<OutlineCheckerOutput>();
    let story = agent("story", &story_model);

    let outline_result = Runner::run(&outline, user_input("Sci-fi please"))
        .await
        .unwrap();
    let checked = Runner::run(&checker, outline_result.to_input_list())
        .await
        .unwrap();
    let decision: OutlineCheckerOutput = checked.final_output_as().unwrap();
    assert!(decision.good_quality && decision.is_scifi);

    let outline_text = outline_result.final_output_text().unwrap();
    let story_result = Runner::run(&story, outline_text.as_str()).await.unwrap();
    assert_eq!(story_result.final_output_text().unwrap(), "Final story");
    assert_eq!(
        story_model.last_call().unwrap().request.input,
        vec![InputItem::user("Outline ready")]
    );
}

/// Python: `test_routing_stream_emits_text_and_updates_inputs`.
#[tokio::test]
async fn a_streamed_reply_flows_through_and_updates_the_input_history() {
    let triage_model = model(vec![text("Bonjour")]);
    let triage = agent("triage_agent", &triage_model);

    let mut streamed = Runner::run_streamed(&triage, "Salut");
    let mut deltas = String::new();
    {
        let mut events = std::pin::pin!(streamed.stream_events());
        while let Some(event) = events.next().await {
            if let Some(delta) = event.unwrap().text_delta() {
                deltas.push_str(delta);
            }
        }
    }
    assert_eq!(deltas, "Bonjour");
    assert_eq!(streamed.final_output_text().as_deref(), Some("Bonjour"));
    let result = streamed.result().expect("complete");
    assert_eq!(result.new_items.len(), 1);
    let input_list = result.to_input_list();
    assert_eq!(input_list.len(), 2);
    assert_eq!(input_list[0], InputItem::user("Salut"));
    let assistant = serde_json::to_value(&input_list[1]).unwrap();
    assert_eq!(assistant["role"], "assistant");
    assert_eq!(assistant["type"], "message");
    assert_eq!(assistant["content"][0]["text"], "Bonjour");
}

#[derive(Deserialize, JsonSchema)]
struct MathHomeworkOutput {
    reasoning: String,
    is_math_homework: bool,
}

/// Python: `test_input_guardrail_agent_trips_and_returns_info`.
#[tokio::test]
async fn an_input_guardrail_backed_by_an_agent_trips_with_its_info() {
    let guardrail_model = model(vec![text(
        &json!({"reasoning": "math detected", "is_math_homework": true}).to_string(),
    )]);
    let guardrail_agent =
        Arc::new(agent("guardrail", &guardrail_model).with_output_type::<MathHomeworkOutput>());
    let math_guardrail = InputGuardrail::new("math", move |ctx, _agent, input: Vec<InputItem>| {
        let guardrail_agent = guardrail_agent.clone();
        async move {
            let result =
                Runner::run_with_options(guardrail_agent, input, RunOptions::with_run_context(ctx))
                    .await?;
            let out: MathHomeworkOutput = result.final_output_as()?;
            Ok(GuardrailFunctionOutput::new(
                json!({"reasoning": out.reasoning, "is_math_homework": out.is_math_homework}),
                out.is_math_homework,
            ))
        }
    });
    let main_model = model(vec![text("Should not run")]);
    let main = agent("main", &main_model).with_input_guardrail(math_guardrail);

    match Runner::run(&main, "Solve 2x+5=11").await {
        Err(AgentsError::InputGuardrailTripwireTriggered {
            guardrail_result, ..
        }) => {
            let info = &guardrail_result.output.output_info;
            assert_eq!(info["is_math_homework"], json!(true));
            assert_eq!(info["reasoning"], json!("math detected"));
        }
        other => panic!("expected an input tripwire, got {other:?}"),
    }
    assert!(main_model.calls().is_empty() || main_model.remaining_steps() == 1);
}

#[derive(Deserialize, JsonSchema)]
struct MessageOutput {
    reasoning: String,
    response: String,
    user_name: Option<String>,
}

/// Python: `test_output_guardrail_blocks_sensitive_data`.
#[tokio::test]
async fn an_output_guardrail_blocks_a_phone_number() {
    let sensitive_data_check = OutputGuardrail::new(
        "sensitive",
        |_ctx, _agent, output: Value| async move {
            let message: MessageOutput = serde_json::from_value(output).unwrap();
            let contains_phone =
                message.response.contains("650") || message.reasoning.contains("650");
            Ok(GuardrailFunctionOutput::new(
                json!({"contains_phone": contains_phone, "user_name_known": message.user_name.is_some()}),
                contains_phone,
            ))
        },
    );
    let m = model(vec![text(
        &json!({
            "reasoning": "User shared phone 650-123-4567",
            "response": "Thanks!",
            "user_name": null,
        })
        .to_string(),
    )]);
    let assistant = agent("Assistant", &m)
        .with_output_type::<MessageOutput>()
        .with_output_guardrail(sensitive_data_check);

    match Runner::run(&assistant, "My phone number is 650-123-4567.").await {
        Err(AgentsError::OutputGuardrailTripwireTriggered {
            guardrail_result, ..
        }) => {
            assert_eq!(
                guardrail_result.output.output_info["contains_phone"],
                json!(true)
            );
        }
        other => panic!("expected an output tripwire, got {other:?}"),
    }
}

/// Python: `test_streaming_guardrail_style_cancel_after_threshold`.
#[tokio::test]
async fn an_immediate_cancel_stops_the_stream_once_a_threshold_is_reached() {
    // The model emits two chunks and then stalls; an immediate cancel must stop the run there.
    let m = model(vec![ScriptedStep::stream_with(|_call| {
        let delta = |text: &str| {
            Ok(json!({
                "type": "response.output_text.delta",
                "item_id": "msg_1",
                "output_index": 0,
                "content_index": 0,
                "delta": text,
            }))
        };
        futures::stream::iter(vec![
            Ok(json!({"type": "response.created", "response": {"id": "resp_1", "status": "in_progress", "output": []}})),
            delta("Chunk1 "),
            delta("Chunk2 "),
        ])
        .chain(futures::stream::pending())
        .boxed()
    })]);
    let talkative = agent("talkative", &m);
    let mut streamed = Runner::run_streamed(&talkative, "Start");
    let handle = streamed.cancel_handle();
    let mut collected = String::new();
    {
        let mut events = std::pin::pin!(streamed.stream_events());
        while let Some(event) = events.next().await {
            let Ok(event) = event else { break };
            if let Some(delta) = event.text_delta() {
                collected.push_str(delta);
                handle.cancel(CancelMode::Immediate);
            }
        }
    }
    assert!(collected.contains("Chunk1"));
    assert!(!collected.contains("Chunk3"));
    assert!(streamed.final_output().is_none());
    assert!(streamed.is_complete());
}

/// Python: `test_streaming_cancel_after_turn_allows_turn_completion`.
#[tokio::test]
async fn an_after_turn_cancel_lets_the_current_turn_finish() {
    let m = model(vec![text("Hello"), text("World")]);
    let talkative = agent("talkative", &m);
    let mut streamed = Runner::run_streamed(&talkative, "Hi");
    let handle = streamed.cancel_handle();
    let mut deltas = String::new();
    {
        let mut events = std::pin::pin!(streamed.stream_events());
        while let Some(event) = events.next().await {
            if let Some(delta) = event.unwrap().text_delta() {
                deltas.push_str(delta);
                handle.cancel(CancelMode::AfterTurn);
            }
        }
    }
    assert!(deltas.starts_with("Hello"));
    assert!(streamed.is_complete());
    assert!(streamed.final_output_text().is_some());
}

/// Python: `test_streaming_handoff_emits_agent_updated_event`.
#[tokio::test]
async fn a_streamed_handoff_emits_an_agent_updated_event() {
    let delegate_model = model(vec![text("delegate reply")]);
    let delegate = Arc::new(agent("delegate", &delegate_model));
    let triage_model = model(vec![calls(vec![
        assistant_message("triage summary"),
        tool_call("transfer_to_delegate", json!({}), "handoff_call"),
    ])]);
    let triage = agent("triage", &triage_model).with_handoffs([handoff(delegate.clone())]);

    let mut streamed = Runner::run_streamed(&triage, "Help me");
    let mut updates = Vec::new();
    {
        let mut events = std::pin::pin!(streamed.stream_events());
        while let Some(event) = events.next().await {
            if let openai_agents_sdk::stream_events::StreamEvent::AgentUpdated { new_agent } =
                event.unwrap()
            {
                updates.push(new_agent.name.clone());
            }
        }
    }
    assert_eq!(
        streamed.final_output_text().as_deref(),
        Some("delegate reply")
    );
    assert_eq!(streamed.result().unwrap().last_agent.name, "delegate");
    assert!(updates.iter().any(|name| name == "delegate"), "{updates:?}");
}

/// Python: `test_forcing_tool_use_behaviors_align_with_example`.
#[tokio::test]
async fn tool_use_behaviors_decide_whether_the_model_runs_again() {
    fn weather_tool() -> FunctionTool<()> {
        #[derive(Deserialize, JsonSchema)]
        struct Args {
            city: String,
        }
        FunctionTool::<()>::new("get_weather", "Weather", |_ctx, args: Args| async move {
            Ok(format!("{}: Sunny", args.city))
        })
    }

    // default: run the model again after the tool
    let default_model = model(vec![
        calls(vec![tool_call(
            "get_weather",
            json!({"city": "Tokyo"}),
            "c1",
        )]),
        text("Done after tool"),
    ]);
    let default_agent = agent("default", &default_model)
        .with_tool(weather_tool())
        .with_tool_use_behavior(ToolUseBehavior::RunLlmAgain);
    let result = Runner::run(&default_agent, "Weather?").await.unwrap();
    assert_eq!(result.final_output_text().unwrap(), "Done after tool");
    assert_eq!(result.raw_responses.len(), 2);

    // stop_on_first_tool: the first tool result is the final output
    let first_model = model(vec![calls(vec![tool_call(
        "get_weather",
        json!({"city": "Paris"}),
        "c2",
    )])]);
    let first_agent = agent("first", &first_model)
        .with_tool(weather_tool())
        .with_tool_use_behavior(ToolUseBehavior::StopOnFirstTool)
        .with_model_settings(ModelSettings {
            tool_choice: Some(ToolChoice::required()),
            ..Default::default()
        });
    let result = Runner::run(&first_agent, "Weather?").await.unwrap();
    assert_eq!(result.final_output_text().unwrap(), "Paris: Sunny");
    assert_eq!(result.raw_responses.len(), 1);

    // custom: formats the output itself
    let custom_model = model(vec![calls(vec![tool_call(
        "get_weather",
        json!({"city": "Berlin"}),
        "c3",
    )])]);
    let custom_agent = agent("custom", &custom_model)
        .with_tool(weather_tool())
        .with_tool_use_behavior(ToolUseBehavior::Custom(Arc::new(|_ctx, results| {
            Box::pin(async move {
                Ok(ToolsToFinalOutputResult::final_output(format!(
                    "Custom:{}",
                    results[0].output.to_model_string()
                )))
            })
        })));
    let result = Runner::run(&custom_agent, "Weather?").await.unwrap();
    assert_eq!(result.final_output_text().unwrap(), "Custom:Berlin: Sunny");
}

/// Python: `test_routing_multi_turn_continues_with_handoff_agent`.
#[tokio::test]
async fn a_follow_up_turn_continues_with_the_handoff_agent() {
    let delegate_model = model(vec![text("Bonjour")]);
    let delegate = Arc::new(agent("delegate", &delegate_model));
    let triage_model = model(vec![
        calls(vec![tool_call("transfer_to_delegate", json!({}), "h1")]),
        text("handoff completed"),
    ]);
    let triage = agent("triage", &triage_model).with_handoffs([handoff(delegate.clone())]);

    let first = Runner::run(&triage, "Help me in French").await.unwrap();
    assert_eq!(first.final_output_text().unwrap(), "Bonjour");
    assert_eq!(first.last_agent.name, "delegate");

    delegate_model.enqueue(text("Encore?")).unwrap();
    let mut follow_up = first.to_input_list();
    follow_up.push(InputItem::user("Encore!"));
    let second = Runner::run(&*delegate, follow_up.clone()).await.unwrap();
    assert_eq!(second.final_output_text().unwrap(), "Encore?");
    assert_eq!(delegate_model.last_call().unwrap().request.input, follow_up);
}

#[derive(Clone)]
struct AppContext {
    language_preference: String,
}

/// Python: `test_agents_as_tools_conditional_enabling_matches_preference`.
#[tokio::test]
async fn only_the_enabled_agent_tools_are_invoked_for_a_preference() {
    let scenarios: [(&str, &[&str]); 3] = [
        ("spanish_only", &["respond_spanish"]),
        ("french_spanish", &["respond_french", "respond_spanish"]),
        (
            "european",
            &["respond_french", "respond_italian", "respond_spanish"],
        ),
    ];
    for (preference, expected_tools) in scenarios {
        let spanish_model = model(vec![text("ES hola")]);
        let french_model = model(vec![text("FR bonjour")]);
        let italian_model = model(vec![text("IT ciao")]);
        let spanish =
            Agent::<AppContext>::new("spanish").with_model_instance(spanish_model.clone());
        let french = Agent::<AppContext>::new("french").with_model_instance(french_model.clone());
        let italian =
            Agent::<AppContext>::new("italian").with_model_instance(italian_model.clone());

        let tool_calls = expected_tools
            .iter()
            .map(|name| tool_call(name, json!({"input": "Hi"}), &format!("call_{name}")))
            .collect();
        let orchestrator_model = model(vec![calls(tool_calls), text("Done")]);

        let french_enabled = openai_agents_sdk::tool::IsEnabled::Dynamic(Arc::new(
            |ctx: RunContext<AppContext>, _agent: String| {
                let preference = ctx.context().language_preference.clone();
                Box::pin(async move { preference == "french_spanish" || preference == "european" })
            },
        ));
        let italian_enabled = openai_agents_sdk::tool::IsEnabled::Dynamic(Arc::new(
            |ctx: RunContext<AppContext>, _agent: String| {
                let preference = ctx.context().language_preference.clone();
                Box::pin(async move { preference == "european" })
            },
        ));
        let orchestrator = Agent::<AppContext>::new("orchestrator")
            .with_model_instance(orchestrator_model.clone())
            .with_tool(spanish.as_tool("respond_spanish", "Spanish"))
            .with_tool(
                french.as_tool_with(
                    openai_agents_sdk::agent::AgentAsToolOptions::new()
                        .tool_name("respond_french")
                        .tool_description("French")
                        .is_enabled(french_enabled),
                ),
            )
            .with_tool(
                italian.as_tool_with(
                    openai_agents_sdk::agent::AgentAsToolOptions::new()
                        .tool_name("respond_italian")
                        .tool_description("Italian")
                        .is_enabled(italian_enabled),
                ),
            )
            .with_model_settings(ModelSettings {
                tool_choice: Some(ToolChoice::required()),
                ..Default::default()
            });

        let context = AppContext {
            language_preference: preference.to_string(),
        };
        let result = Runner::run_with_options(&orchestrator, "Hello", RunOptions::new(context))
            .await
            .unwrap();
        assert_eq!(result.final_output_text().unwrap(), "Done", "{preference}");
        for (name, model) in [
            ("respond_spanish", &spanish_model),
            ("respond_french", &french_model),
            ("respond_italian", &italian_model),
        ] {
            assert_eq!(
                !model.calls().is_empty(),
                expected_tools.contains(&name),
                "{preference}: {name}"
            );
        }
    }
}

/// Python: `test_agents_as_tools_orchestrator_runs_multiple_translations`.
#[tokio::test]
async fn an_orchestrator_calls_several_translation_agents_then_summarizes() {
    let spanish_model = model(vec![text("ES hola")]);
    let french_model = model(vec![text("FR bonjour")]);
    let spanish = agent("spanish", &spanish_model);
    let french = agent("french", &french_model);
    let orchestrator_model = model(vec![
        calls(vec![tool_call(
            "translate_to_spanish",
            json!({"input": "Hi"}),
            "translate_spanish",
        )]),
        calls(vec![tool_call(
            "translate_to_french",
            json!({"input": "Hi"}),
            "translate_french",
        )]),
        text("Summary complete"),
    ]);
    let orchestrator = agent("orchestrator", &orchestrator_model)
        .with_tool(spanish.as_tool("translate_to_spanish", "Spanish"))
        .with_tool(french.as_tool("translate_to_french", "French"));

    let result = Runner::run(&orchestrator, "Hi").await.unwrap();
    assert_eq!(result.final_output_text().unwrap(), "Summary complete");
    assert_eq!(
        spanish_model.last_call().unwrap().request.input,
        vec![InputItem::user("Hi")]
    );
    assert_eq!(
        french_model.last_call().unwrap().request.input,
        vec![InputItem::user("Hi")]
    );
    assert_eq!(result.raw_responses.len(), 3);
}

/// Python: `test_agent_as_tool_streaming_example_collects_events`. The nested run streams its
/// events to `on_stream`; every event names the nested agent and the outer tool call.
#[tokio::test]
async fn an_agent_tool_streams_nested_events_to_its_callback() {
    let billing_model = model(vec![text("Billing: $100")]);
    let billing = agent("billing", &billing_model);
    let received: Arc<Mutex<Vec<(String, String)>>> = Arc::new(Mutex::new(Vec::new()));
    let sink = received.clone();
    let on_stream = move |event: openai_agents_sdk::agent::AgentToolStreamEvent<()>| {
        let sink = sink.clone();
        async move {
            let tool = event.tool_call.as_ref().map(|c| c.name.clone());
            sink.lock()
                .unwrap()
                .push((event.agent.name.clone(), tool.unwrap_or_default()));
        }
    };
    let tool = billing.as_tool_with(
        openai_agents_sdk::agent::AgentAsToolOptions::new()
            .tool_name("billing_agent")
            .tool_description("Answer billing questions")
            .on_stream(on_stream),
    );
    let main_model = model(vec![
        calls(vec![tool_call(
            "billing_agent",
            json!({"input": "Need bill"}),
            "b1",
        )]),
        text("Final answer"),
    ]);
    let support = agent("support", &main_model)
        .with_tool(tool)
        .with_model_settings(ModelSettings {
            tool_choice: Some(ToolChoice::required()),
            ..Default::default()
        });

    let result = Runner::run(&support, "How much is my bill?").await.unwrap();
    assert_eq!(result.final_output_text().unwrap(), "Final answer");
    let received = received.lock().unwrap();
    assert!(
        !received.is_empty(),
        "on_stream should capture nested events"
    );
    assert!(
        received
            .iter()
            .all(|(agent, tool)| agent == "billing" && tool == "billing_agent")
    );
}

#[derive(Deserialize, JsonSchema)]
struct NoArgs {}

fn tool_outputs(input: &[InputItem]) -> Vec<(String, String)> {
    input
        .iter()
        .filter_map(|item| {
            let value = serde_json::to_value(item).ok()?;
            (value["type"] == "function_call_output").then(|| {
                (
                    value["call_id"].as_str().unwrap_or_default().to_string(),
                    value["output"].as_str().unwrap_or_default().to_string(),
                )
            })
        })
        .collect()
}

/// Python: `test_agents_as_tools_streaming_subagent_cancellation_preserves_parent_output`. A
/// tool of the nested agent fails: the nested model sees the default failure message beside its
/// sibling's output, finishes, and the outer turn keeps both sub-agent results.
#[tokio::test]
async fn a_failing_nested_tool_does_not_cost_the_parent_its_sibling_outputs() {
    let events = Arc::new(Mutex::new(0usize));
    let sink = events.clone();
    let on_stream = move |_event: openai_agents_sdk::agent::AgentToolStreamEvent<()>| {
        let sink = sink.clone();
        async move {
            *sink.lock().unwrap() += 1;
        }
    };
    let status_model = model(vec![text("Status: ok")]);
    let status = agent("status", &status_model);
    let ok_tool = FunctionTool::<()>::new("ok_tool", "ok", |_ctx, _args: NoArgs| async {
        Ok("Investigation: ok".to_string())
    });
    let failing_tool =
        FunctionTool::<()>::new("cancel_tool", "fails", |_ctx, _args: NoArgs| async {
            Err::<String, _>(AgentsError::user("tool-cancelled"))
        });
    let observability_model = model(vec![
        calls(vec![
            tool_call("ok_tool", json!({}), "inner_ok"),
            tool_call("cancel_tool", json!({}), "inner_cancel"),
        ]),
        text("Nested summary"),
    ]);
    let observability = agent("observability", &observability_model)
        .with_tool(ok_tool)
        .with_tool(failing_tool)
        .with_model_settings(ModelSettings {
            tool_choice: Some(ToolChoice::required()),
            ..Default::default()
        });
    let orchestrator_model = model(vec![
        calls(vec![
            tool_call("status_agent", json!({"input": "Hi"}), "outer_status"),
            tool_call(
                "observability_agent",
                json!({"input": "Hi"}),
                "outer_observability",
            ),
        ]),
        text("Summary complete"),
    ]);
    let orchestrator = agent("orchestrator", &orchestrator_model)
        .with_tool(status.as_tool("status_agent", "Status"))
        .with_tool(
            observability.as_tool_with(
                openai_agents_sdk::agent::AgentAsToolOptions::new()
                    .tool_name("observability_agent")
                    .tool_description("Observability")
                    .on_stream(on_stream),
            ),
        )
        .with_model_settings(ModelSettings {
            tool_choice: Some(ToolChoice::required()),
            ..Default::default()
        });

    let result = Runner::run(&orchestrator, "Hi").await.unwrap();
    assert_eq!(result.final_output_text().unwrap(), "Summary complete");
    assert_eq!(result.raw_responses.len(), 2);
    assert!(*events.lock().unwrap() > 0, "the nested run must stream");
    assert_eq!(
        status_model.last_call().unwrap().request.input,
        vec![InputItem::user("Hi")]
    );

    let nested = tool_outputs(&observability_model.last_call().unwrap().request.input);
    assert_eq!(
        nested,
        vec![
            ("inner_ok".to_string(), "Investigation: ok".to_string()),
            (
                "inner_cancel".to_string(),
                "An error occurred while running the tool. Please try again.".to_string()
            ),
        ]
    );
    let outer = tool_outputs(&orchestrator_model.last_call().unwrap().request.input);
    assert_eq!(
        outer,
        vec![
            ("outer_status".to_string(), "Status: ok".to_string()),
            (
                "outer_observability".to_string(),
                "Nested summary".to_string()
            ),
        ]
    );
}
