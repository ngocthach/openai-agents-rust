//! Port of `test_invalid_final_output_handler.py`: the `invalid_final_output` run error handler
//! recovers a structured-output turn that returned invalid or no JSON, in blocking and streamed
//! runs.

use std::sync::{Arc, Mutex};

use async_trait::async_trait;
use futures::StreamExt;
use openai_agents_sdk::debug::{reset_dont_log_model_data, set_dont_log_model_data};
use openai_agents_sdk::guardrail::GuardrailFunctionOutput;
use openai_agents_sdk::lifecycle::RunHooks;
use openai_agents_sdk::result::RunResult;
use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_error_handlers::{
    RunErrorHandlerInput, RunErrorHandlerResult, RunErrorHandlers,
};
use openai_agents_sdk::stream_events::{RunItemStreamEventName, StreamEvent};
use openai_agents_sdk::testing::{
    ScriptedModel, ScriptedStep, assistant_message, function_call_with_id,
};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{
    Agent, AgentsError, InMemorySession, ItemHelpers, OutputGuardrail, Result, RunContext,
    RunItemKind, Runner, Session,
};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Debug, Deserialize, JsonSchema)]
#[allow(dead_code)]
struct FinalOutput {
    summary: String,
}

fn fallback() -> Value {
    json!({"summary": "safe fallback"})
}

fn scripted(steps: Vec<Vec<openai_agents_sdk::items::OutputItem>>) -> Arc<ScriptedModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(
        ScriptedModel::with_steps(steps.into_iter().map(ScriptedStep::output).collect()).unwrap(),
    )
}

fn agent(model: &Arc<ScriptedModel>) -> Agent<()> {
    Agent::<()>::new("test")
        .with_model_instance(model.clone())
        .with_output_type::<FinalOutput>()
}

type Handler = Box<dyn Fn(RunErrorHandlerInput<()>) -> Option<RunErrorHandlerResult> + Send + Sync>;

fn handlers(handler: Handler) -> RunErrorHandlers<()> {
    let handler = Arc::new(handler);
    RunErrorHandlers::new().on_invalid_final_output(move |data| {
        let handler = handler.clone();
        async move { Ok(handler(data)) }
    })
}

fn fallback_handler() -> RunErrorHandlers<()> {
    handlers(Box::new(|_| Some(RunErrorHandlerResult::new(fallback()))))
}

struct Outcome {
    result: RunResult<()>,
    fallback_events: usize,
}

/// Run blocking or streamed and collect the `message_output_created` events that carry the
/// fallback text.
async fn run(agent: Agent<()>, options: RunOptions<()>, streamed: bool) -> Result<Outcome> {
    if !streamed {
        let result = Runner::run_with_options(&agent, "user_message", options).await?;
        return Ok(Outcome {
            result,
            fallback_events: 0,
        });
    }
    let mut stream = Runner::run_streamed_with_options(agent, "user_message", options);
    let mut fallback_events = 0;
    {
        let events = stream.stream_events();
        futures::pin_mut!(events);
        while let Some(event) = events.next().await {
            if let StreamEvent::RunItem {
                name: RunItemStreamEventName::MessageOutputCreated,
                item,
            } = event?
                && ItemHelpers::text_message_output(&item) == r#"{"summary":"safe fallback"}"#
            {
                fallback_events += 1;
            }
        }
    }
    Ok(Outcome {
        result: stream
            .into_result()
            .expect("a finished stream has a result"),
        fallback_events,
    })
}

fn message_texts(result: &RunResult<()>) -> Vec<String> {
    result
        .new_items
        .iter()
        .filter(|item| matches!(item.kind, RunItemKind::MessageOutput { .. }))
        .map(ItemHelpers::text_message_output)
        .collect()
}

fn message_error(err: &AgentsError) -> &str {
    match err {
        AgentsError::ModelBehaviorError { message, .. } => message,
        other => panic!("expected ModelBehaviorError, got {other:?}"),
    }
}

/// Python: `test_invalid_final_output_raises_without_handler`.
#[tokio::test]
async fn invalid_final_output_raises_without_a_handler() {
    let model = scripted(vec![vec![assistant_message("not valid json")]]);
    let err = Runner::run(&agent(&model), "user_message")
        .await
        .unwrap_err();
    assert!(message_error(&err).contains("Invalid JSON"), "{err}");
}

/// Python: `test_invalid_final_output_handler_returns_validated_fallback`.
#[tokio::test]
async fn handler_returns_a_validated_fallback() {
    let model = scripted(vec![vec![assistant_message("not valid json")]]);
    let result = Runner::run_with_options(
        &agent(&model),
        "user_message",
        RunOptions::new(()).error_handlers(handlers(Box::new(|data| {
            assert!(matches!(
                *data.error,
                AgentsError::ModelBehaviorError { .. }
            ));
            assert!(!data.run_data.raw_responses.is_empty());
            assert_eq!(
                ItemHelpers::text_message_outputs(&data.run_data.new_items),
                "not valid json"
            );
            Some(RunErrorHandlerResult::new(fallback()))
        }))),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output, Some(fallback()));
    assert_eq!(
        message_texts(&result),
        ["not valid json", r#"{"summary":"safe fallback"}"#]
    );
}

/// Python: `test_invalid_final_output_handler_can_skip_fallback_history`.
#[tokio::test]
async fn handler_can_skip_the_fallback_history() {
    let model = scripted(vec![vec![assistant_message("not valid json")]]);
    let result = Runner::run_with_options(
        &agent(&model),
        "user_message",
        RunOptions::new(()).error_handlers(handlers(Box::new(|_| {
            Some(RunErrorHandlerResult::new(fallback()).without_history())
        }))),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output, Some(fallback()));
    assert_eq!(
        ItemHelpers::text_message_outputs(&result.new_items),
        "not valid json"
    );
}

/// Python: `test_invalid_final_output_handler_rejects_invalid_fallback` (the pydantic
/// serializer warning is Python-only).
#[tokio::test]
async fn handler_with_an_invalid_fallback_is_rejected() {
    // Python patches `DONT_LOG_MODEL_DATA` off so the error text is not redacted.
    set_dont_log_model_data(false);
    let model = scripted(vec![vec![assistant_message("not valid json")]]);
    let err = Runner::run_with_options(
        &agent(&model),
        "user_message",
        RunOptions::new(()).error_handlers(handlers(Box::new(|_| {
            Some(RunErrorHandlerResult::new(json!({"unexpected": "value"})))
        }))),
    )
    .await
    .unwrap_err();
    reset_dont_log_model_data();
    assert!(matches!(err, AgentsError::UserError { .. }), "{err:?}");
    assert!(
        err.to_string()
            .contains("Invalid run error handler final_output"),
        "{err}"
    );
}

/// Python: `test_invalid_final_output_handler_can_decline_recovery`.
#[tokio::test]
async fn handler_can_decline_recovery() {
    let model = scripted(vec![vec![assistant_message("not valid json")]]);
    let err = Runner::run_with_options(
        &agent(&model),
        "user_message",
        RunOptions::new(()).error_handlers(handlers(Box::new(|_| None))),
    )
    .await
    .unwrap_err();
    assert!(message_error(&err).contains("Invalid JSON"), "{err}");
}

/// Python: `test_invalid_final_output_handler_does_not_catch_other_model_behavior_errors`.
#[tokio::test]
async fn handler_does_not_catch_other_model_behavior_errors() {
    let model = scripted(vec![vec![function_call_with_id(
        "missing_tool",
        "{}",
        "call_1",
    )]]);
    let called = Arc::new(Mutex::new(false));
    let flag = called.clone();
    let err = Runner::run_with_options(
        &agent(&model),
        "user_message",
        RunOptions::new(()).error_handlers(handlers(Box::new(move |_| {
            *flag.lock().unwrap() = true;
            Some(RunErrorHandlerResult::new(fallback()))
        }))),
    )
    .await
    .unwrap_err();
    assert!(message_error(&err).contains("not found"), "{err}");
    assert!(!*called.lock().unwrap());
}

/// Python: `test_empty_structured_output_handler_avoids_another_model_turn`
/// (`[]` and `[get_text_message("")]`).
#[tokio::test]
async fn empty_structured_output_handler_avoids_another_model_turn() {
    for invalid in [vec![], vec![assistant_message("")]] {
        let model = scripted(vec![
            invalid,
            vec![assistant_message(r#"{"summary":"unused"}"#)],
        ]);
        let result = Runner::run_with_options(
            &agent(&model),
            "user_message",
            RunOptions::new(()).error_handlers(handlers(Box::new(|data| {
                assert_eq!(
                    message_error(&data.error),
                    "Model returned no final output for the structured output type."
                );
                Some(RunErrorHandlerResult::new(fallback()))
            }))),
        )
        .await
        .unwrap();
        assert_eq!(result.final_output, Some(fallback()));
        assert_eq!(model.remaining_steps(), 1);
    }
}

/// Python: `test_empty_structured_output_without_fallback_keeps_existing_next_turn_behavior`
/// (no handlers, and a handler that declines).
#[tokio::test]
async fn empty_structured_output_without_a_fallback_runs_the_next_turn() {
    for error_handlers in [None, Some(handlers(Box::new(|_| None)))] {
        let model = scripted(vec![
            vec![],
            vec![assistant_message(r#"{"summary":"second turn"}"#)],
        ]);
        let mut options = RunOptions::new(());
        if let Some(h) = error_handlers {
            options = options.error_handlers(h);
        }
        let result = Runner::run_with_options(&agent(&model), "user_message", options)
            .await
            .unwrap();
        assert_eq!(result.final_output, Some(json!({"summary": "second turn"})));
        assert_eq!(model.remaining_steps(), 0);
    }
}

/// Python: `test_streamed_invalid_final_output_emits_exact_fallback_item`.
#[tokio::test]
async fn streamed_invalid_final_output_emits_the_exact_fallback_item() {
    let model = scripted(vec![vec![assistant_message("not valid json")]]);
    let session = Arc::new(InMemorySession::new("invalid-final-output"));
    let outcome = run(
        agent(&model),
        RunOptions::new(())
            .session(session.clone())
            .error_handlers(fallback_handler()),
        true,
    )
    .await
    .unwrap();
    assert_eq!(outcome.result.final_output, Some(fallback()));
    assert_eq!(outcome.fallback_events, 1);
    assert_eq!(
        message_texts(&outcome.result),
        ["not valid json", r#"{"summary":"safe fallback"}"#]
    );
    let stored: Vec<String> = session
        .get_items(None)
        .await
        .unwrap()
        .iter()
        .filter(|item| item.item_type() == Some("message"))
        .filter(|item| item.0["role"] == "assistant")
        .map(|item| {
            item.0["content"][0]["text"]
                .as_str()
                .unwrap_or_default()
                .to_string()
        })
        .collect();
    assert_eq!(stored, ["not valid json", r#"{"summary":"safe fallback"}"#]);
}

/// Python: `test_streamed_empty_structured_output_handler_avoids_another_model_turn`.
#[tokio::test]
async fn streamed_empty_structured_output_handler_avoids_another_model_turn() {
    let model = scripted(vec![
        vec![],
        vec![assistant_message(r#"{"summary":"unused"}"#)],
    ]);
    let outcome = run(
        agent(&model),
        RunOptions::new(()).error_handlers(fallback_handler()),
        true,
    )
    .await
    .unwrap();
    assert_eq!(outcome.result.final_output, Some(fallback()));
    assert_eq!(model.remaining_steps(), 1);
    assert_eq!(outcome.fallback_events, 1);
}

#[derive(Default)]
struct RecordingRunHooks {
    final_outputs: Mutex<Vec<Value>>,
}

#[async_trait]
impl RunHooks<()> for RecordingRunHooks {
    async fn on_agent_end(&self, _context: &RunContext<()>, _agent: &Agent<()>, output: &Value) {
        self.final_outputs.lock().unwrap().push(output.clone());
    }
}

/// Python: `test_invalid_final_output_fallback_runs_hooks_and_output_guardrails`.
#[tokio::test]
async fn fallback_runs_hooks_and_output_guardrails() {
    let model = scripted(vec![vec![assistant_message("not valid json")]]);
    let hooks = Arc::new(RecordingRunHooks::default());
    let guarded = Arc::new(Mutex::new(Vec::<Value>::new()));
    let record = guarded.clone();
    let agent = agent(&model).with_output_guardrail(OutputGuardrail::new(
        "record_output",
        move |_ctx, _agent, output: Value| {
            let record = record.clone();
            async move {
                record.lock().unwrap().push(output);
                Ok(GuardrailFunctionOutput::pass())
            }
        },
    ));
    let result = Runner::run_with_options(
        &agent,
        "user_message",
        RunOptions::new(())
            .hooks(hooks.clone())
            .error_handlers(fallback_handler()),
    )
    .await
    .unwrap();
    assert_eq!(result.final_output, Some(fallback()));
    assert_eq!(*hooks.final_outputs.lock().unwrap(), vec![fallback()]);
    assert_eq!(*guarded.lock().unwrap(), vec![fallback()]);
    assert_eq!(result.output_guardrail_results.len(), 1);
}

#[derive(Deserialize, JsonSchema)]
struct SideEffectArgs {
    value: String,
}

/// Python: `test_invalid_final_output_fallback_does_not_retry_or_replay_tools`.
#[tokio::test]
async fn fallback_does_not_retry_or_replay_tools() {
    for streamed in [false, true] {
        let side_effects = Arc::new(Mutex::new(Vec::<String>::new()));
        let log = side_effects.clone();
        let tool = FunctionTool::new(
            "record_side_effect",
            "records a value",
            move |_ctx, args: SideEffectArgs| {
                let log = log.clone();
                async move {
                    log.lock().unwrap().push(args.value.clone());
                    Ok(format!("recorded:{}", args.value))
                }
            },
        );
        let model = scripted(vec![
            vec![function_call_with_id(
                "record_side_effect",
                json!({"value": "once"}).to_string(),
                "first_call",
            )],
            vec![assistant_message("not valid json")],
            vec![function_call_with_id(
                "record_side_effect",
                json!({"value": "replayed"}).to_string(),
                "replayed_call",
            )],
            vec![assistant_message(r#"{"summary":"unexpected retry"}"#)],
        ]);
        let agent = agent(&model).with_tool(tool);
        let outcome = run(
            agent,
            RunOptions::new(()).error_handlers(fallback_handler()),
            streamed,
        )
        .await
        .unwrap();
        assert_eq!(outcome.result.final_output, Some(fallback()));
        if streamed {
            assert_eq!(outcome.fallback_events, 1);
        }
        assert_eq!(*side_effects.lock().unwrap(), vec!["once".to_string()]);
        assert_eq!(model.remaining_steps(), 2);
    }
}
