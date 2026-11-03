//! Rust side of the payload-parity harness. Each test mirrors `parity/scenarios/<name>/py.py`
//! and runs only under `parity/run.sh capture <name>`, which points `OPENAI_BASE_URL` at the
//! local capture server and sets `OPENAI_AGENTS_PARITY_CAPTURE=1`; the server records the
//! request, so the tests only have to build the same agent as the Python script and run it.

use std::collections::HashMap;
use std::sync::Arc;

use futures::StreamExt;
#[cfg(feature = "sqlite")]
use openai_agents_sdk::Session;
use openai_agents_sdk::extensions::handoff_prompt::prompt_with_handoff_instructions;
#[cfg(feature = "sqlite")]
use openai_agents_sdk::memory::SqliteSession;
use openai_agents_sdk::stream_events::StreamEvent;
use openai_agents_sdk::{
    Agent, AgentsError, FunctionTool, GuardrailFunctionOutput, InputGuardrail, InputItem,
    ModelSettings, OutputGuardrail, Reasoning, RunOptions, Runner, ToolChoice, handoff,
};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

fn enabled() -> bool {
    std::env::var("OPENAI_AGENTS_PARITY_CAPTURE").as_deref() == Ok("1")
}

/// The hello-world agent with no model set: the request carries the default model and
/// default settings.
#[tokio::test]
async fn default_model() {
    if !enabled() {
        return;
    }
    let agent = Agent::<()>::new("Assistant").with_instructions("You only respond in haikus.");
    let result = Runner::run(&agent, "Tell me about recursion in programming.")
        .await
        .unwrap();
    println!("{}", result.final_output_text().unwrap_or_default());
}

/// `model = "gpt-6-luna"`: a name no default-settings pattern matches, so the request must
/// carry no implicit reasoning or verbosity settings.
#[tokio::test]
async fn explicit_gpt6_luna() {
    if !enabled() {
        return;
    }
    let agent = Agent::<()>::new("Assistant")
        .with_instructions("You only respond in haikus.")
        .with_model("gpt-6-luna");
    let result = Runner::run(&agent, "Tell me about recursion in programming.")
        .await
        .unwrap();
    println!("{}", result.final_output_text().unwrap_or_default());
}

#[derive(Deserialize, JsonSchema)]
struct WeatherArgs {
    /// The city to get the weather for
    city: String,
}

#[derive(Serialize)]
struct Weather {
    city: String,
    temperature_range: String,
    conditions: String,
}

/// One typed function tool, a scripted call and a final answer: the second request must
/// carry the function call and its output.
#[tokio::test]
async fn function_tools() {
    if !enabled() {
        return;
    }
    let get_weather = FunctionTool::new(
        "get_weather",
        "Get the current weather information for a specified city.",
        |_ctx, args: WeatherArgs| async move {
            let w = Weather {
                city: args.city,
                temperature_range: "14-20C".into(),
                conditions: "Sunny with wind.".into(),
            };
            Ok(serde_json::to_value(w)?)
        },
    );
    let agent = Agent::<()>::new("Hello world")
        .with_instructions("You are a helpful agent.")
        .with_tool(get_weather);
    let result = Runner::run(&agent, "What's the weather in Tokyo?")
        .await
        .unwrap();
    println!("{}", result.final_output_text().unwrap_or_default());
}

/// Every model setting both SDKs expose, on a model with no implicit defaults.
#[tokio::test]
async fn model_settings() {
    if !enabled() {
        return;
    }
    let settings = ModelSettings {
        temperature: Some(0.3),
        top_p: Some(0.9),
        frequency_penalty: Some(0.1),
        presence_penalty: Some(0.2),
        tool_choice: Some(ToolChoice::auto()),
        parallel_tool_calls: Some(false),
        truncation: Some("auto".into()),
        max_tokens: Some(256),
        reasoning: Some(Reasoning {
            effort: Some("low".into()),
            summary: Some("auto".into()),
            ..Default::default()
        }),
        verbosity: Some("low".into()),
        metadata: Some(HashMap::from([("run".to_string(), "parity".to_string())])),
        store: Some(false),
        prompt_cache_retention: Some("in_memory".into()),
        include_usage: Some(true),
        response_include: Some(vec!["reasoning.encrypted_content".into()]),
        top_logprobs: Some(2),
        extra_query: Some(HashMap::from([("q".to_string(), "1".to_string())])),
        extra_body: Some(serde_json::Map::from_iter([(
            "prompt_cache_key".to_string(),
            serde_json::json!("parity"),
        )])),
        extra_headers: Some(HashMap::from([("x-parity".to_string(), "1".to_string())])),
        ..Default::default()
    };
    let agent = Agent::<()>::new("Settings")
        .with_instructions("Reply with exactly PACKAGED_SETTINGS_OK.")
        .with_model("gpt-6-luna")
        .with_model_settings(settings);
    let result = Runner::run(&agent, "Confirm the packaged settings path.")
        .await
        .unwrap();
    println!("{}", result.final_output_text().unwrap_or_default());
}

#[derive(Debug, Deserialize, JsonSchema)]
struct CalendarEvent {
    name: String,
    date: String,
    participants: Vec<String>,
}

/// A typed output becomes a strict `text.format` JSON schema.
#[tokio::test]
async fn structured_output() {
    if !enabled() {
        return;
    }
    let agent = Agent::<()>::new("Calendar extractor")
        .with_instructions("Extract calendar events from text")
        .with_output_type::<CalendarEvent>();
    let result = Runner::run(
        &agent,
        "Alice and Bob are going to a science fair on Friday.",
    )
    .await
    .unwrap();
    let event: CalendarEvent = result.final_output_as().unwrap();
    println!(
        "{} on {} with {}",
        event.name,
        event.date,
        event.participants.join(", ")
    );
}

/// A triage agent with two handoffs and the recommended prompt prefix; the scripted handoff
/// makes the second request go to the history tutor.
#[tokio::test]
async fn handoffs() {
    if !enabled() {
        return;
    }
    let history_tutor = Arc::new(
        Agent::<()>::new("History Tutor")
            .with_handoff_description("Specialist agent for historical questions")
            .with_instructions("You answer history questions clearly and concisely."),
    );
    let math_tutor = Arc::new(
        Agent::<()>::new("Math Tutor")
            .with_handoff_description("Specialist agent for math questions")
            .with_instructions("You explain math step by step and include worked examples."),
    );
    let triage = Agent::<()>::new("Triage Agent")
        .with_instructions(prompt_with_handoff_instructions(
            "Route each homework question to the right specialist.",
        ))
        .with_handoffs([handoff(history_tutor), handoff(math_tutor)]);
    let result = Runner::run(&triage, "Who was the first president of the United States?")
        .await
        .unwrap();
    println!("{}", result.final_output_text().unwrap_or_default());
    println!("Answered by: {}", result.last_agent.name);
}

/// An orchestrator calling a specialist agent as a tool: the scripted call runs the inner
/// agent (its own request) before the orchestrator's second turn.
#[tokio::test]
async fn agents_as_tools() {
    if !enabled() {
        return;
    }
    let spanish = Agent::<()>::new("spanish_agent")
        .with_instructions("You translate the user's message to Spanish");
    let french = Agent::<()>::new("french_agent")
        .with_instructions("You translate the user's message to French");
    let orchestrator = Agent::<()>::new("orchestrator_agent")
        .with_instructions(
            "You are a translation agent. You use the tools given to you to translate. \
             If asked for multiple translations, you call the relevant tools.",
        )
        .with_tool(spanish.as_tool(
            "translate_to_spanish",
            "Translate the user's message to Spanish",
        ))
        .with_tool(french.as_tool(
            "translate_to_french",
            "Translate the user's message to French",
        ));
    let result = Runner::run(&orchestrator, "Say 'Hello, how are you?' in Spanish.")
        .await
        .unwrap();
    println!("{}", result.final_output_text().unwrap_or_default());
}

#[derive(Debug, Deserialize, JsonSchema)]
struct MathHomeworkOutput {
    reasoning: String,
    is_math_homework: bool,
}

/// An agent-based input guardrail run in parallel with the first model call, the same
/// guardrail blocking, and a function output guardrail: the verdict is the set of model
/// requests each configuration sends and the error raised.
#[tokio::test]
async fn guardrails() {
    if !enabled() {
        return;
    }
    let guardrail_agent = Arc::new(
        Agent::<()>::new("Guardrail check")
            .with_instructions("Check if the user is asking you to do their math homework.")
            .with_output_type::<MathHomeworkOutput>(),
    );
    let math_guardrail = |blocking: bool| {
        let guardrail_agent = guardrail_agent.clone();
        let guardrail = InputGuardrail::new(
            "math_homework",
            move |ctx, _agent, input: Vec<InputItem>| {
                let guardrail_agent = guardrail_agent.clone();
                async move {
                    let result = Runner::run_with_options(
                        guardrail_agent,
                        input,
                        RunOptions::with_run_context(ctx),
                    )
                    .await?;
                    let out: MathHomeworkOutput = result.final_output_as()?;
                    Ok(GuardrailFunctionOutput::new(
                        json!({"reasoning": out.reasoning, "is_math_homework": out.is_math_homework}),
                        out.is_math_homework,
                    ))
                }
            },
        );
        if blocking {
            guardrail.blocking()
        } else {
            guardrail
        }
    };
    let no_equations =
        OutputGuardrail::new("no_equations", |_ctx, _agent, output: Value| async move {
            let has_equation = output.as_str().unwrap_or_default().contains('=');
            Ok(GuardrailFunctionOutput::new(
                json!({"has_equation": has_equation}),
                has_equation,
            ))
        });
    let instructions = "You are a customer support agent. You help customers with their questions.";
    let question = "Can you help me solve for x: 2x + 5 = 11";
    let cases = [
        (
            "parallel",
            Agent::<()>::new("Customer support agent")
                .with_instructions(format!("{instructions} (parallel)"))
                .with_input_guardrail(math_guardrail(false)),
        ),
        (
            "blocking",
            Agent::<()>::new("Customer support agent")
                .with_instructions(format!("{instructions} (blocking)"))
                .with_input_guardrail(math_guardrail(true)),
        ),
        (
            "output",
            Agent::<()>::new("Customer support agent")
                .with_instructions(format!("{instructions} (output)"))
                .with_output_guardrail(no_equations),
        ),
    ];
    for (label, agent) in cases {
        match Runner::run(&agent, question).await {
            Ok(result) => println!(
                "{label}: no tripwire: {}",
                result.final_output_text().unwrap_or_default()
            ),
            Err(AgentsError::InputGuardrailTripwireTriggered { .. }) => {
                println!("{label}: input tripwire")
            }
            Err(AgentsError::OutputGuardrailTripwireTriggered { .. }) => {
                println!("{label}: output tripwire")
            }
            Err(e) => panic!("{label}: {e}"),
        }
    }
}

#[derive(Deserialize, JsonSchema)]
struct CityArgs {
    city: String,
}

/// A tool that needs approval for Oakland pauses the run (no second request); approving
/// after a JSON round trip of the state resumes with the tool output, rejecting resumes with
/// the rejection text.
#[tokio::test]
async fn human_in_the_loop() {
    if !enabled() {
        return;
    }
    let get_temperature =
        FunctionTool::new(
            "get_temperature",
            "Get the temperature",
            |_ctx, a: CityArgs| async move {
                Ok(format!("The temperature in {} is 20° Celsius", a.city))
            },
        )
        .with_needs_approval_fn(|_ctx, args, _call_id| async move {
            args["city"]
                .as_str()
                .unwrap_or_default()
                .contains("Oakland")
        });
    let agent = Agent::<()>::new("Weather assistant")
        .with_instructions("Answer weather questions with the provided tools.")
        .with_tool(get_temperature);
    for approve in [true, false] {
        let mut result = Runner::run(&agent, "What is the temperature in Oakland?")
            .await
            .unwrap();
        while !result.interruptions.is_empty() {
            let stored = result.to_state().to_json_string().unwrap();
            let mut state = openai_agents_sdk::RunState::from_json_str(&stored).unwrap();
            for item in state.get_interruptions() {
                if approve {
                    state.approve(&item, false);
                } else {
                    state.reject(&item, false, None);
                }
            }
            result = Runner::run(&agent, state).await.unwrap();
        }
        println!("{}", result.final_output_text().unwrap_or_default());
    }
}

/// Three turns through a SQLite session (each request carries the stored history plus the
/// new message), two turns chained with `previous_response_id`, and one run with
/// `conversation_id`.
#[cfg(feature = "sqlite")]
#[tokio::test]
async fn sessions() {
    if !enabled() {
        return;
    }
    let out = std::path::PathBuf::from(std::env::var("PARITY_OUT").unwrap());
    let agent = Agent::<()>::new("Assistant").with_instructions("Reply very concisely.");
    let session: Arc<dyn Session> =
        Arc::new(SqliteSession::open("conversation_123", out.join("rs_session.db")).unwrap());
    for question in [
        "What city is the Golden Gate Bridge in?",
        "What state is it in?",
        "What's the population of that state?",
    ] {
        let result = Runner::run_with_options(
            &agent,
            question,
            RunOptions::new(()).session(session.clone()),
        )
        .await
        .unwrap();
        println!(
            "Assistant: {}",
            result.final_output_text().unwrap_or_default()
        );
    }

    let concise = Agent::<()>::new("Assistant")
        .with_instructions("You are a helpful assistant. be VERY concise.");
    let result = Runner::run(&concise, "What is the largest country in South America?")
        .await
        .unwrap();
    println!("{}", result.final_output_text().unwrap_or_default());
    let result = Runner::run_with_options(
        &concise,
        "What is the capital of that country?",
        RunOptions::new(()).previous_response_id(result.last_response_id().unwrap()),
    )
    .await
    .unwrap();
    println!("{}", result.final_output_text().unwrap_or_default());

    let helper = Agent::<()>::new("Assistant").with_instructions("You are a helpful assistant.");
    let result = Runner::run_with_options(
        &helper,
        "Hello",
        RunOptions::new(()).conversation_id("conv_parity_123"),
    )
    .await
    .unwrap();
    println!("{}", result.final_output_text().unwrap_or_default());
}

/// A streamed run with one tool call: the request bodies carry `stream: true`, and the
/// sequence of events the SDK emits from the same SSE is written to `$PARITY_OUT/rs.events.json`
/// in the shared vocabulary (`raw:<api event type>`, `item:<run item event name>`,
/// `agent_updated:<agent name>`).
#[tokio::test]
async fn streaming() {
    if !enabled() {
        return;
    }
    let how_many_jokes = FunctionTool::new_no_args(
        "how_many_jokes",
        "Return a number of jokes.",
        |_ctx| async { Ok(3) },
    );
    let agent = Agent::<()>::new("Joker")
        .with_instructions("First call the `how_many_jokes` tool, then tell that many jokes.")
        .with_tool(how_many_jokes);
    let mut result = Runner::run_streamed(&agent, "Hello");
    let mut events = Vec::new();
    {
        let mut stream = std::pin::pin!(result.stream_events());
        while let Some(event) = stream.next().await {
            events.push(match event.unwrap() {
                StreamEvent::RawResponse(raw) => {
                    format!("raw:{}", raw["type"].as_str().unwrap_or_default())
                }
                StreamEvent::RunItem { name, .. } => format!("item:{}", name.as_str()),
                StreamEvent::AgentUpdated { new_agent } => {
                    format!("agent_updated:{}", new_agent.name)
                }
            });
        }
    }
    println!("{}", result.final_output_text().unwrap_or_default());
    let out = std::path::PathBuf::from(std::env::var("PARITY_OUT").unwrap());
    std::fs::write(
        out.join("rs.events.json"),
        serde_json::to_string_pretty(&events).unwrap(),
    )
    .unwrap();
}

/// Every hosted tool config both SDKs expose, as pure request payload.
#[tokio::test]
async fn hosted_tools() {
    if !enabled() {
        return;
    }
    use openai_agents_sdk::tool::{
        CodeInterpreterTool, FileSearchTool, HostedMCPTool, ImageGenerationTool, WebSearchTool,
    };
    let agent = Agent::<()>::new("Hosted tools")
        .with_instructions("Use the hosted tools when they help.")
        .with_model("gpt-6-luna")
        .with_tool(WebSearchTool {
            user_location: Some(json!({"type": "approximate", "city": "Tokyo"})),
            filters: Some(json!({"allowed_domains": ["openai.com"]})),
            search_context_size: Some("low".into()),
            ..Default::default()
        })
        .with_tool(FileSearchTool {
            vector_store_ids: vec!["vs_parity".into()],
            max_num_results: Some(3),
            include_search_results: true,
            ..Default::default()
        })
        .with_tool(CodeInterpreterTool::new())
        .with_tool(ImageGenerationTool::new())
        .with_tool(HostedMCPTool::<()>::new(json!({
            "type": "mcp",
            "server_label": "deepwiki",
            "server_url": "https://mcp.deepwiki.com/mcp",
            "require_approval": "never",
        })));
    let result = Runner::run(&agent, "What's new in the openai-agents SDK?")
        .await
        .unwrap();
    println!("{}", result.final_output_text().unwrap_or_default());
}

/// A remote image, an inline base64 image, a remote PDF and an inline PDF as user content
/// parts, then a text question.
#[tokio::test]
async fn inputs_media() {
    if !enabled() {
        return;
    }
    let image_url = "https://upload.wikimedia.org/wikipedia/commons/0/0c/GoldenGateBridge-001.jpg";
    let pdf_url = "https://www.berkshirehathaway.com/letters/2024ltr.pdf";
    let b64 = "aGVsbG8=";
    let agent = Agent::<()>::new("Assistant").with_instructions("You are a helpful assistant.");
    let input = vec![
        InputItem::user_parts(vec![
            json!({"type": "input_image", "detail": "auto", "image_url": image_url}),
        ]),
        InputItem::user_parts(vec![json!({
            "type": "input_image", "detail": "auto",
            "image_url": format!("data:image/jpeg;base64,{b64}"),
        })]),
        InputItem::user_parts(vec![json!({"type": "input_file", "file_url": pdf_url})]),
        InputItem::user_parts(vec![json!({
            "type": "input_file",
            "file_data": format!("data:application/pdf;base64,{b64}"),
            "filename": "letter.pdf",
        })]),
        InputItem::user("What do you see in these?"),
    ];
    let result = Runner::run(&agent, input).await.unwrap();
    println!("{}", result.final_output_text().unwrap_or_default());
}

/// A Responses prompt template and instructions computed at run time.
#[tokio::test]
async fn prompts_dynamic() {
    if !enabled() {
        return;
    }
    let poet = Agent::<()>::new("Poet").with_prompt(
        openai_agents_sdk::Prompt::new("pmpt_parity")
            .with_version("1")
            .with_variable("poem_style", "haiku"),
    );
    let result = Runner::run(&poet, "Write a poem about recursion.")
        .await
        .unwrap();
    println!("{}", result.final_output_text().unwrap_or_default());

    let name = "Sora";
    let greeter =
        Agent::<()>::new("Greeter").with_dynamic_instructions(move |_ctx, _agent| async move {
            Ok(format!("The user's name is {name}. Greet them."))
        });
    let result = Runner::run(&greeter, "Hi").await.unwrap();
    println!("{}", result.final_output_text().unwrap_or_default());
}

#[derive(Deserialize, JsonSchema)]
struct CityOnly {
    /// The city to get the weather for
    city: String,
}

/// The weather tool with a plain-string result, shared by the chat, tracing and reasoning
/// scenarios.
fn weather_string_tool() -> FunctionTool<()> {
    FunctionTool::new(
        "get_weather",
        "Get the current weather information for a specified city.",
        |_ctx, args: CityOnly| async move { Ok(format!("Sunny in {}", args.city)) },
    )
}

/// The same tool-calling agent through the Chat Completions model.
#[tokio::test]
async fn chat_completions() {
    if !enabled() {
        return;
    }
    use openai_agents_sdk::{OpenAIChatCompletionsModel, OpenAIClient};
    let model = OpenAIChatCompletionsModel::new("gpt-6-luna", OpenAIClient::from_env());
    let agent = Agent::<()>::new("Hello world")
        .with_instructions("You are a helpful agent.")
        .with_tool(weather_string_tool())
        .with_model_instance(Arc::new(model));
    let result = Runner::run(&agent, "What's the weather in Tokyo?")
        .await
        .unwrap();
    println!("{}", result.final_output_text().unwrap_or_default());
}

/// The first model request gets a 429; the SDK-level retry sends it again.
#[tokio::test]
async fn retry() {
    if !enabled() {
        return;
    }
    use openai_agents_sdk::model_settings::{ModelRetryBackoffSettings, ModelRetrySettings};
    let settings = ModelSettings {
        retry: Some(ModelRetrySettings {
            max_retries: Some(1),
            backoff: Some(ModelRetryBackoffSettings {
                initial_delay: Some(0.01),
                max_delay: Some(0.02),
                jitter: Some(false),
                ..Default::default()
            }),
            retry_statuses: Some(vec![429]),
            ..Default::default()
        }),
        ..Default::default()
    };
    let agent = Agent::<()>::new("Assistant")
        .with_instructions("Reply with exactly RETRY_OK.")
        .with_model("gpt-6-luna")
        .with_model_settings(settings);
    let result = Runner::run(&agent, "Confirm the retry path.")
        .await
        .unwrap();
    println!("{}", result.final_output_text().unwrap_or_default());
}

/// A tool-calling run with the trace exporter pointed at the capture server.
#[tokio::test]
async fn tracing_export() {
    if !enabled() {
        return;
    }
    use openai_agents_sdk::tracing::{
        BackendSpanExporter, BatchTraceProcessor, flush_traces_async, set_trace_processors,
    };
    let endpoint = format!(
        "{}/traces/ingest",
        std::env::var("OPENAI_BASE_URL").unwrap()
    );
    set_trace_processors(vec![Arc::new(BatchTraceProcessor::new(Arc::new(
        BackendSpanExporter::new().with_endpoint(endpoint),
    )))]);
    let agent = Agent::<()>::new("Hello world")
        .with_instructions("You are a helpful agent.")
        .with_model("gpt-6-luna")
        .with_tool(weather_string_tool());
    let result = Runner::run(&agent, "What's the weather in Tokyo?")
        .await
        .unwrap();
    println!("{}", result.final_output_text().unwrap_or_default());
    flush_traces_async().await;
}

/// The model returns a reasoning item before a tool call; the second request must send that
/// reasoning item back along with the call and its output.
#[tokio::test]
async fn reasoning_items() {
    if !enabled() {
        return;
    }
    let agent = Agent::<()>::new("Hello world")
        .with_instructions("You are a helpful agent.")
        .with_model("gpt-6-luna")
        .with_model_settings(ModelSettings {
            reasoning: Some(Reasoning {
                effort: Some("low".into()),
                summary: Some("auto".into()),
                ..Default::default()
            }),
            ..Default::default()
        })
        .with_tool(weather_string_tool());
    let result = Runner::run(&agent, "What's the weather in Tokyo?")
        .await
        .unwrap();
    println!("{}", result.final_output_text().unwrap_or_default());
}
