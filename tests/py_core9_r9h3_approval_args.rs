//! Ports of the portable cases `tests/test_function_tool_approval_arguments.py` left behind:
//! application-supplied defaults and constraint checks run only after an approval, and an
//! input guardrail that blocks an approved call keeps the failure policy out of the way.

use std::sync::{Arc, Mutex};

use openai_agents_sdk::items::{RunInput, RunItemKind};
use openai_agents_sdk::testing::{FakeModel, assistant_message, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::tool_guardrails::{ToolGuardrailFunctionOutput, ToolInputGuardrail};
use openai_agents_sdk::{Agent, RunResult, RunState, Runner};
use serde::Deserialize;

type Log<T> = Arc<Mutex<Vec<T>>>;

fn log<T>() -> Log<T> {
    Arc::new(Mutex::new(Vec::new()))
}

fn scripted_agent(tool: FunctionTool<()>, arguments: &str) -> Agent<()> {
    let model = Arc::new(FakeModel::new());
    model.add_turns(vec![
        vec![function_call_with_id(
            tool.name.clone(),
            arguments,
            "call_test",
        )],
        vec![assistant_message("done")],
    ]);
    Agent::<()>::new("test")
        .with_model_instance(model)
        .with_tool(tool)
}

async fn run_agent(
    agent: &Agent<()>,
    input: impl Into<RunInput>,
    streamed: bool,
) -> openai_agents_sdk::Result<RunResult<()>> {
    if streamed {
        let mut stream = Runner::run_streamed(agent, input);
        stream.finish().await?;
        Ok(stream.into_result().expect("streamed result"))
    } else {
        Runner::run(agent, input).await
    }
}

fn roundtrip(state: &RunState) -> RunState {
    RunState::from_json_str(&state.to_json_string().unwrap()).unwrap()
}

fn outputs(result: &RunResult<()>) -> Vec<String> {
    result
        .new_items
        .iter()
        .filter_map(|item| match &item.kind {
            RunItemKind::ToolCallOutput { output, .. } => Some(
                output
                    .as_str()
                    .map(str::to_string)
                    .unwrap_or_else(|| output.to_string()),
            ),
            _ => None,
        })
        .collect()
}

static FACTORY_EVENTS: Mutex<Vec<&'static str>> = Mutex::new(Vec::new());

fn factory_events() -> std::sync::MutexGuard<'static, Vec<&'static str>> {
    FACTORY_EVENTS.lock().unwrap_or_else(|e| e.into_inner())
}

fn application_default() -> String {
    factory_events().push("validate");
    "safe".into()
}

#[derive(Deserialize, schemars::JsonSchema)]
struct FactoryArgs {
    #[serde(default = "application_default")]
    value: String,
}

/// Python: `test_application_validation_waits_for_explicit_approval[factory]` (streamed x
/// approve). A default produced by application code is not produced while the call waits for a
/// decision, only after it is approved. (The other kinds put user code in a pydantic validator.)
#[tokio::test]
async fn an_application_default_runs_only_after_approval() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for streamed in [false, true] {
        for approve in [false, true] {
            let label = format!("streamed={streamed} approve={approve}");
            factory_events().clear();
            let tool =
                FunctionTool::<()>::new("operation", "", |_ctx, args: FactoryArgs| async move {
                    factory_events().push("body");
                    Ok(args.value)
                })
                .with_needs_approval_fn(|_ctx, _args, _id| async move {
                    factory_events().push("policy");
                    true
                });
            // Generating the schema calls the default function once to record `default`.
            factory_events().clear();
            let agent = scripted_agent(tool, "{}");
            let paused = run_agent(&agent, "go", streamed).await.unwrap();
            assert!(!paused.interruptions.is_empty(), "{label}");
            assert!(
                factory_events().is_empty(),
                "{label}: {:?}",
                factory_events()
            );
            let mut state = roundtrip(&paused.to_state());
            let pending = state.get_interruptions()[0].clone();
            if approve {
                state.approve(&pending, false);
            } else {
                state.reject(&pending, false, None);
            }
            let result = run_agent(&agent, state, streamed).await.unwrap();
            assert!(result.interruptions.is_empty(), "{label}");
            let want: &[&str] = if approve { &["validate", "body"] } else { &[] };
            assert_eq!(*factory_events(), want, "{label}");
        }
    }
}

#[derive(Deserialize, schemars::JsonSchema)]
struct PatternArgs {
    #[schemars(inner(regex(pattern = "^a+$")))]
    value: Vec<String>,
}

/// Python: `test_pattern_validation_runs_after_approval` (the string-pattern half; Rust has no
/// compiled-regex object). A value that breaks the declared pattern is reported through the
/// failure policy only after the approval, and a matching one runs the body.
#[tokio::test]
async fn pattern_validation_runs_after_approval() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for value in ["aaa", "a!"] {
        let events = log::<&'static str>();
        let ev = events.clone();
        let failed = events.clone();
        let tool = FunctionTool::<()>::new("operation", "", move |_ctx, args: PatternArgs| {
            let ev = ev.clone();
            async move {
                ev.lock().unwrap().push("body");
                Ok(args.value[0].clone())
            }
        })
        .with_needs_approval_fn(|_ctx, _args, _id| async { true })
        .with_failure_error_function(move |_ctx, _error| {
            failed.lock().unwrap().push("error");
            "invalid value".to_string()
        });
        let agent = scripted_agent(tool, &serde_json::json!({"value": [value]}).to_string());
        let paused = run_agent(&agent, "go", false).await.unwrap();
        assert!(!paused.interruptions.is_empty(), "{value}");
        assert!(events.lock().unwrap().is_empty(), "{value}");
        let mut state = paused.to_state();
        let pending = state.get_interruptions()[0].clone();
        state.approve(&pending, false);
        let result = run_agent(&agent, state, false).await.unwrap();
        assert!(result.interruptions.is_empty(), "{value}");
        let (want_events, want_output): (&[&str], &str) = if value == "aaa" {
            (&["body"], "aaa")
        } else {
            (&["error"], "invalid value")
        };
        assert_eq!(*events.lock().unwrap(), want_events, "{value}");
        assert_eq!(outputs(&result), [want_output], "{value}");
    }
}

#[derive(Deserialize, schemars::JsonSchema)]
struct ValueArgs {
    value: i64,
}

/// Python: `test_plugin_validation_follows_input_guardrails` (the ordering, without the pydantic
/// plugin events). After approval the input guardrail runs before the arguments are checked: a
/// blocking guardrail answers with its own message and the failure policy never runs; an allowing
/// one lets the invalid value reach the policy.
#[tokio::test]
async fn an_input_guardrail_runs_before_the_failure_policy() {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    for block in [false, true] {
        let events = log::<&'static str>();
        let ev = events.clone();
        let guardrail = ToolInputGuardrail::<()>::new("guardrail", move |_data| {
            let ev = ev.clone();
            async move {
                ev.lock().unwrap().push("guardrail");
                Ok(if block {
                    ToolGuardrailFunctionOutput::reject_content("blocked")
                } else {
                    ToolGuardrailFunctionOutput::allow()
                })
            }
        });
        let ev = events.clone();
        let formatter_events = events.clone();
        let tool = FunctionTool::<()>::new("operation", "", move |_ctx, args: ValueArgs| {
            let ev = ev.clone();
            async move {
                ev.lock().unwrap().push("body");
                Ok(args.value.to_string())
            }
        })
        .with_input_guardrail(guardrail)
        .with_needs_approval_fn(|_ctx, _args, _id| async { true })
        .with_failure_error_function(move |_ctx, _error| {
            formatter_events.lock().unwrap().push("formatter");
            "invalid value".to_string()
        });
        let agent = scripted_agent(tool, r#"{"value":"bad"}"#);
        let paused = run_agent(&agent, "go", false).await.unwrap();
        assert!(!paused.interruptions.is_empty(), "block={block}");
        assert!(events.lock().unwrap().is_empty(), "block={block}");
        let mut state = paused.to_state();
        let pending = state.get_interruptions()[0].clone();
        state.approve(&pending, false);
        let result = run_agent(&agent, state, false).await.unwrap();
        assert!(result.interruptions.is_empty(), "block={block}");
        let want: &[&str] = if block {
            &["guardrail"]
        } else {
            &["guardrail", "formatter"]
        };
        assert_eq!(*events.lock().unwrap(), want, "block={block}");
        let want_output = if block { "blocked" } else { "invalid value" };
        assert_eq!(outputs(&result), [want_output], "block={block}");
    }
}
