//! Rust side of the live parity runs. Each test mirrors `parity/scenarios/live/<name>/py.py`,
//! runs only under `parity/run.sh live <name>` (`OPENAI_AGENTS_PARITY_LIVE=1`, real key and
//! `PARITY_MODEL` in the environment) and writes the same behaviour-shape record as the Python
//! side to `$PARITY_OUT/rs.json`; `parity/compare_live.py` checks both against `expect.json`.

use std::sync::{Arc, Mutex};
use std::time::Instant;

use futures::StreamExt;
#[cfg(feature = "sqlite")]
use openai_agents_sdk::Session;
use openai_agents_sdk::extensions::handoff_prompt::prompt_with_handoff_instructions;
#[cfg(feature = "sqlite")]
use openai_agents_sdk::memory::SqliteSession;
use openai_agents_sdk::stream_events::StreamEvent;
use openai_agents_sdk::tool::WebSearchTool;
use openai_agents_sdk::{
    Agent, AgentsError, FunctionTool, GuardrailFunctionOutput, InputGuardrail, InputItem,
    ModelSettings, Reasoning, RunOptions, RunResult, RunState, Runner,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Map, Value, json};

fn enabled() -> bool {
    std::env::var("OPENAI_AGENTS_PARITY_LIVE").as_deref() == Ok("1")
}

fn model() -> String {
    std::env::var("PARITY_MODEL").unwrap()
}

fn settings(max_tokens: u32) -> ModelSettings {
    ModelSettings {
        max_tokens: Some(max_tokens),
        ..Default::default()
    }
}

/// The behaviour-shape record, with the same keys as `parity/shape.py`.
struct Shape {
    started: Instant,
    record: Map<String, Value>,
}

impl Shape {
    fn new(scenario: &str) -> Self {
        let mut record = Map::new();
        record.insert("scenario".into(), json!(scenario));
        record.insert("sdk".into(), json!("rs"));
        record.insert("model".into(), json!(model()));
        record.insert("error".into(), Value::Null);
        Shape {
            started: Instant::now(),
            record,
        }
    }

    fn result(&mut self, result: &RunResult<()>) {
        let final_output = result.final_output.clone().unwrap_or(Value::Null);
        let parsed = match &final_output {
            Value::Object(_) | Value::Array(_) => final_output.clone(),
            _ => Value::Null,
        };
        let text = match &final_output {
            Value::String(s) => s.clone(),
            Value::Null => String::new(),
            other => other.to_string(),
        };
        let mut tool_calls = Vec::new();
        let mut handoffs = Vec::new();
        let mut annotations = 0usize;
        for item in &result.new_items {
            match &item.kind {
                openai_agents_sdk::items::RunItemKind::ToolCall { raw_item } => {
                    let name = raw_item
                        .get("name")
                        .or_else(|| raw_item.get("type"))
                        .cloned()
                        .unwrap_or(json!("unknown"));
                    let arguments = raw_item
                        .get("arguments")
                        .and_then(Value::as_str)
                        .and_then(|a| serde_json::from_str::<Value>(a).ok())
                        .unwrap_or(Value::Null);
                    tool_calls.push(json!({"name": name, "arguments": arguments}));
                }
                openai_agents_sdk::items::RunItemKind::HandoffOutput {
                    source_agent,
                    target_agent,
                    ..
                } => handoffs.push(format!("{source_agent}->{target_agent}")),
                openai_agents_sdk::items::RunItemKind::MessageOutput { raw_item } => {
                    if let Some(parts) =
                        raw_item.to_value().get("content").and_then(Value::as_array)
                    {
                        for part in parts {
                            annotations += part
                                .get("annotations")
                                .and_then(Value::as_array)
                                .map_or(0, Vec::len);
                        }
                    }
                }
                _ => {}
            }
        }
        let usage = result.usage();
        self.record.insert("final_output".into(), json!(text));
        self.record.insert("final_output_parsed".into(), parsed);
        self.record
            .insert("last_agent".into(), json!(result.last_agent.name));
        self.record.insert(
            "new_item_types".into(),
            json!(
                result
                    .new_items
                    .iter()
                    .map(|i| i.item_type())
                    .collect::<Vec<_>>()
            ),
        );
        self.record.insert("tool_calls".into(), json!(tool_calls));
        self.record.insert("handoffs".into(), json!(handoffs));
        self.record.insert("annotations".into(), json!(annotations));
        self.record.insert(
            "usage".into(),
            json!({
                "requests": usage.requests,
                "input_tokens": usage.input_tokens,
                "output_tokens": usage.output_tokens,
            }),
        );
        self.record
            .insert("response_ids".into(), json!(result.raw_responses.len()));
        self.record
            .insert("input_list_len".into(), json!(result.to_input_list().len()));
    }

    fn error(&mut self, error: &AgentsError) {
        let (kind, status) = match error {
            AgentsError::InputGuardrailTripwireTriggered { .. } => {
                ("InputGuardrailTripwireTriggered", None)
            }
            AgentsError::OutputGuardrailTripwireTriggered { .. } => {
                ("OutputGuardrailTripwireTriggered", None)
            }
            AgentsError::ApiError { status, .. } => ("APIStatusError", Some(*status)),
            _ => ("AgentsError", None),
        };
        self.record.insert(
            "error".into(),
            json!({"type": kind, "status": status, "message": error.to_string().chars().take(200).collect::<String>()}),
        );
    }

    fn extra(&mut self, key: &str, value: Value) {
        self.record.insert(key.into(), value);
    }

    fn write(mut self) {
        self.record.insert(
            "elapsed_ms".into(),
            json!(self.started.elapsed().as_millis() as u64),
        );
        let out = std::path::PathBuf::from(std::env::var("PARITY_OUT").unwrap());
        std::fs::write(
            out.join("rs.json"),
            serde_json::to_string_pretty(&Value::Object(self.record.clone())).unwrap(),
        )
        .unwrap();
        println!(
            "{}",
            json!({
                "scenario": self.record["scenario"],
                "error": self.record["error"],
                "usage": self.record.get("usage").cloned().unwrap_or(Value::Null),
                "elapsed_ms": self.record["elapsed_ms"],
            })
        );
    }
}

/// Run and record: the result on success, the error otherwise.
async fn record(
    shape: &mut Shape,
    run: impl std::future::Future<Output = Result<RunResult<()>, AgentsError>>,
) -> Option<RunResult<()>> {
    match run.await {
        Ok(result) => {
            shape.result(&result);
            Some(result)
        }
        Err(error) => {
            shape.error(&error);
            None
        }
    }
}

#[derive(Deserialize, JsonSchema)]
struct CityArgs {
    /// The city to get the weather for
    city: String,
}

#[tokio::test]
async fn hello() {
    if !enabled() {
        return;
    }
    let mut shape = Shape::new("hello");
    let agent = Agent::<()>::new("Assistant")
        .with_instructions("You only respond in haikus.")
        .with_model(model())
        .with_model_settings(settings(400));
    record(
        &mut shape,
        Runner::run(&agent, "Tell me about recursion in programming."),
    )
    .await;
    shape.write();
}

#[tokio::test]
async fn tools_two_turn() {
    if !enabled() {
        return;
    }
    let mut shape = Shape::new("tools_two_turn");
    let get_weather = FunctionTool::new(
        "get_weather",
        "Get the current weather information for a specified city.",
        |_ctx, args: CityArgs| async move { Ok(format!("Sunny in {}, 14-20C, light wind.", args.city)) },
    );
    let agent = Agent::<()>::new("Weather agent")
        .with_instructions(
            "Always use the get_weather tool to answer weather questions. Be concise.",
        )
        .with_model(model())
        .with_model_settings(settings(400))
        .with_tool(get_weather);
    record(
        &mut shape,
        Runner::run(&agent, "What's the weather in Tokyo?"),
    )
    .await;
    shape.write();
}

#[derive(Debug, Deserialize, JsonSchema)]
struct CalendarEvent {
    name: String,
    date: String,
    participants: Vec<String>,
}

#[tokio::test]
async fn structured_output() {
    if !enabled() {
        return;
    }
    let mut shape = Shape::new("structured_output");
    let agent = Agent::<()>::new("Calendar extractor")
        .with_instructions("Extract calendar events from text")
        .with_model(model())
        .with_model_settings(settings(400))
        .with_output_type::<CalendarEvent>();
    if let Some(result) = record(
        &mut shape,
        Runner::run(
            &agent,
            "Alice and Bob are going to a science fair on Friday.",
        ),
    )
    .await
        && let Ok(event) = result.final_output_as::<CalendarEvent>()
    {
        {
            let mut participants: Vec<String> = event
                .participants
                .iter()
                .map(|p| p.to_lowercase())
                .collect();
            participants.sort();
            shape.extra("participants_sorted", json!(participants));
            shape.extra("event", json!({"name": event.name, "date": event.date}));
        }
    }
    shape.write();
}

#[tokio::test]
async fn handoff() {
    if !enabled() {
        return;
    }
    let mut shape = Shape::new("handoff");
    let spanish = Arc::new(
        Agent::<()>::new("Spanish Assistant")
            .with_handoff_description(
                "A Spanish-speaking assistant for messages written in Spanish.",
            )
            .with_instructions("You only speak Spanish and are extremely concise.")
            .with_model(model())
            .with_model_settings(settings(400)),
    );
    let english = Arc::new(
        Agent::<()>::new("English Assistant")
            .with_handoff_description(
                "An English-speaking assistant for messages written in English.",
            )
            .with_instructions("You only speak English and are extremely concise.")
            .with_model(model())
            .with_model_settings(settings(400)),
    );
    let triage = Agent::<()>::new("Triage")
        .with_instructions(prompt_with_handoff_instructions(
            "Hand off every message to the assistant that speaks its language. Never answer yourself.",
        ))
        .with_model(model())
        .with_model_settings(settings(400))
        .with_handoffs([
            openai_agents_sdk::handoff(spanish),
            openai_agents_sdk::handoff(english),
        ]);
    record(
        &mut shape,
        Runner::run(&triage, "Hola, ¿cómo estás? Responde en una frase."),
    )
    .await;
    shape.write();
}

#[tokio::test]
async fn streaming() {
    if !enabled() {
        return;
    }
    let mut shape = Shape::new("streaming");
    let how_many_jokes = FunctionTool::new_no_args(
        "how_many_jokes",
        "Return the number of jokes to tell.",
        |_ctx| async { Ok(2) },
    );
    let agent = Agent::<()>::new("Joker")
        .with_instructions(
            "First call the `how_many_jokes` tool, then tell that many very short jokes.",
        )
        .with_model(model())
        .with_model_settings(settings(400))
        .with_tool(how_many_jokes);
    let mut stream = Runner::run_streamed(&agent, "Hello");
    let mut events = Vec::new();
    let mut failure = None;
    {
        let mut stream_events = std::pin::pin!(stream.stream_events());
        while let Some(event) = stream_events.next().await {
            match event {
                Ok(StreamEvent::RawResponse(raw)) => {
                    events.push(format!("raw:{}", raw["type"].as_str().unwrap_or_default()))
                }
                Ok(StreamEvent::RunItem { name, .. }) => {
                    events.push(format!("item:{}", name.as_str()))
                }
                Ok(StreamEvent::AgentUpdated { new_agent }) => {
                    events.push(format!("agent_updated:{}", new_agent.name))
                }
                Err(e) => {
                    failure = Some(e);
                    break;
                }
            }
        }
    }
    let item_events: Vec<&String> = events.iter().filter(|e| !e.starts_with("raw:")).collect();
    shape.extra("stream_item_events", json!(item_events));
    shape.extra("stream_event_types", json!(events));
    match (failure, stream.into_result()) {
        (Some(e), _) => shape.error(&e),
        (None, Some(result)) => shape.result(&result),
        (None, None) => shape.extra(
            "error",
            json!({"type": "NoResult", "status": null, "message": "stream ended without a result"}),
        ),
    }
    shape.write();
}

#[tokio::test]
async fn approval_resume() {
    if !enabled() {
        return;
    }
    let mut shape = Shape::new("approval_resume");
    let runs = Arc::new(Mutex::new(0usize));
    let counter = runs.clone();
    let get_temperature = FunctionTool::new(
        "get_temperature",
        "Get the temperature for a city.",
        move |_ctx, args: CityArgs| {
            let counter = counter.clone();
            async move {
                *counter.lock().unwrap() += 1;
                Ok(format!("The temperature in {} is 20° Celsius", args.city))
            }
        },
    )
    .with_needs_approval(true);
    let agent = Agent::<()>::new("Weather assistant")
        .with_instructions(
            "Always use the get_temperature tool to answer temperature questions. Be concise.",
        )
        .with_model(model())
        .with_model_settings(settings(400))
        .with_tool(get_temperature);
    let mut interruptions = Vec::new();
    let outcome: Result<RunResult<()>, AgentsError> = async {
        let mut result = Runner::run(&agent, "What is the temperature in Oakland?").await?;
        interruptions.push(result.interruptions.len());
        while !result.interruptions.is_empty() {
            let stored = result.to_state().to_json_string()?;
            let mut state = RunState::from_json_str(&stored)?;
            for item in state.get_interruptions() {
                state.approve(&item, false);
            }
            result = Runner::run(&agent, state).await?;
            interruptions.push(result.interruptions.len());
        }
        Ok(result)
    }
    .await;
    match outcome {
        Ok(result) => shape.result(&result),
        Err(e) => shape.error(&e),
    }
    shape.extra("interruptions", json!(interruptions));
    shape.extra("tool_runs", json!(*runs.lock().unwrap()));
    shape.write();
}

#[cfg(feature = "sqlite")]
#[tokio::test]
async fn session_two_turns() {
    if !enabled() {
        return;
    }
    let mut shape = Shape::new("session_two_turns");
    let out = std::path::PathBuf::from(std::env::var("PARITY_OUT").unwrap());
    let agent = Agent::<()>::new("Assistant")
        .with_instructions("Reply very concisely.")
        .with_model(model())
        .with_model_settings(settings(400));
    let session: Arc<dyn Session> =
        Arc::new(SqliteSession::open("live_123", out.join("rs_session.db")).unwrap());
    let outcome: Result<RunResult<()>, AgentsError> = async {
        Runner::run_with_options(
            &agent,
            "My favourite colour is turquoise. Just acknowledge it.",
            RunOptions::new(()).session(session.clone()),
        )
        .await?;
        Runner::run_with_options(
            &agent,
            "What is my favourite colour? Answer with the colour only.",
            RunOptions::new(()).session(session.clone()),
        )
        .await
    }
    .await;
    match outcome {
        Ok(result) => {
            let text = result
                .final_output_text()
                .unwrap_or_default()
                .to_lowercase();
            shape.result(&result);
            shape.extra("mentions_fact", json!(text.contains("turquoise")));
            let items = session.get_items(None).await.unwrap();
            shape.extra("session_items", json!(items.len()));
        }
        Err(e) => shape.error(&e),
    }
    shape.write();
}

#[tokio::test]
async fn model_settings_reasoning() {
    if !enabled() {
        return;
    }
    let mut shape = Shape::new("model_settings_reasoning");
    let agent = Agent::<()>::new("Assistant")
        .with_instructions("Answer in one short sentence.")
        .with_model(model())
        .with_model_settings(ModelSettings {
            max_tokens: Some(400),
            reasoning: Some(Reasoning {
                effort: Some("low".into()),
                summary: Some("auto".into()),
                ..Default::default()
            }),
            verbosity: Some("low".into()),
            ..Default::default()
        });
    if let Some(result) = record(&mut shape, Runner::run(&agent, "Why is the sky blue?")).await {
        let has_reasoning = result
            .new_items
            .iter()
            .any(|i| i.item_type() == "reasoning_item");
        shape.extra("has_reasoning_item", json!(has_reasoning));
    }
    shape.write();
}

#[tokio::test]
async fn agents_as_tools() {
    if !enabled() {
        return;
    }
    let mut shape = Shape::new("agents_as_tools");
    let spanish = Agent::<()>::new("spanish_agent")
        .with_instructions(
            "You translate the user's message to Spanish. Reply with the translation only.",
        )
        .with_model(model())
        .with_model_settings(settings(400));
    let orchestrator = Agent::<()>::new("orchestrator_agent")
        .with_instructions(
            "You are a translation agent. You never translate yourself: always call the \
             translate_to_spanish tool, then reply with its result only.",
        )
        .with_model(model())
        .with_model_settings(settings(400))
        .with_tool(spanish.as_tool(
            "translate_to_spanish",
            "Translate the user's message to Spanish",
        ));
    record(
        &mut shape,
        Runner::run(&orchestrator, "Translate to Spanish: Good morning, friend."),
    )
    .await;
    shape.write();
}

#[tokio::test]
async fn web_search() {
    if !enabled() {
        return;
    }
    let mut shape = Shape::new("web_search");
    let agent = Agent::<()>::new("Researcher")
        .with_instructions("Use web search to answer, then reply in one sentence with a citation.")
        .with_model(model())
        .with_model_settings(settings(600))
        .with_tool(WebSearchTool {
            search_context_size: Some("low".into()),
            ..Default::default()
        });
    if let Some(result) = record(
        &mut shape,
        Runner::run(
            &agent,
            "What is the current stable version of the Rust compiler?",
        ),
    )
    .await
    {
        let calls = result
            .new_items
            .iter()
            .filter(|i| i.item_type() == "tool_call_item")
            .count();
        shape.extra("search_calls", json!(calls));
    }
    shape.write();
}

#[derive(Debug, Deserialize, JsonSchema)]
struct MathHomeworkOutput {
    reasoning: String,
    is_math_homework: bool,
}

#[tokio::test]
async fn guardrail_tripwire() {
    if !enabled() {
        return;
    }
    let mut shape = Shape::new("guardrail_tripwire");
    let guardrail_agent = Arc::new(
        Agent::<()>::new("Guardrail check")
            .with_instructions("Check if the user is asking you to do their math homework.")
            .with_model(model())
            .with_model_settings(settings(400))
            .with_output_type::<MathHomeworkOutput>(),
    );
    let math_guardrail = InputGuardrail::new(
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
    )
    .blocking();
    let agent = Agent::<()>::new("Customer support agent")
        .with_instructions(
            "You are a customer support agent. You help customers with their questions.",
        )
        .with_model(model())
        .with_model_settings(settings(400))
        .with_input_guardrail(math_guardrail);
    let tripped = match Runner::run(&agent, "Can you help me solve for x: 2x + 5 = 11").await {
        Ok(result) => {
            shape.result(&result);
            false
        }
        Err(e) => {
            let tripped = matches!(e, AgentsError::InputGuardrailTripwireTriggered { .. });
            shape.error(&e);
            tripped
        }
    };
    shape.extra("tripped", json!(tripped));
    shape.write();
}

#[tokio::test]
async fn unsupported_param() {
    if !enabled() {
        return;
    }
    let mut shape = Shape::new("unsupported_param");
    let agent = Agent::<()>::new("Assistant")
        .with_instructions("Answer in one word.")
        .with_model(model())
        .with_model_settings(ModelSettings {
            max_tokens: Some(400),
            temperature: Some(0.2),
            ..Default::default()
        });
    let outcome = match Runner::run(&agent, "What colour is the sky on a clear day?").await {
        Ok(result) => {
            shape.result(&result);
            "ok".to_string()
        }
        Err(e) => {
            let status = match &e {
                AgentsError::ApiError { status, .. } => status.to_string(),
                _ => "None".to_string(),
            };
            shape.error(&e);
            format!("error:{status}")
        }
    };
    shape.extra("outcome", json!(outcome));
    shape.write();
}
