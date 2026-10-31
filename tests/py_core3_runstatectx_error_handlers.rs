//! Port of `tests/test_run_internal_error_handlers.py`, run through the public runner.
//!
//! Python tests call the private `run_internal.error_handlers` helpers directly. The same
//! behaviour is observable through `RunOptions::error_handlers`: a max-turns handler's output is
//! validated against the agent's output type, and a handler that declines leaves the error.
//!
//! Python-only:
//! - `format_final_output_text` falling back to `str(value)` for an arbitrary Python object, and
//!   the monkeypatched custom output schema: Rust handler output is always a JSON value.
//! - `validate_handler_final_output` raising for a set (not JSON): a Rust `Value` cannot hold one.
//! - a handler result carrying an unknown key: `RunErrorHandlerResult` is a typed struct.

use std::sync::Arc;

use openai_agents_sdk::run::RunOptions;
use openai_agents_sdk::run_config::RunConfig;
use openai_agents_sdk::run_error_handlers::{RunErrorHandlerResult, RunErrorHandlers};
use openai_agents_sdk::testing::{FakeModel, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, AgentsError, Runner};
use serde_json::{Value, json};

/// A model that always asks for the tool, so a one-turn budget is exceeded on the second turn.
fn looping_agent(max_turns_model: &Arc<FakeModel>) -> Agent<()> {
    max_turns_model.add_turn(vec![function_call_with_id("ping", "{}", "call-1")]);
    max_turns_model.add_turn(vec![function_call_with_id("ping", "{}", "call-2")]);
    let ping =
        FunctionTool::<()>::new_no_args("ping", "Ping", |_c| async { Ok("pong".to_string()) });
    Agent::<()>::new("max-turns")
        .with_model_instance(max_turns_model.clone())
        .with_tools(vec![ping])
}

fn options_with(handlers: RunErrorHandlers<()>) -> RunOptions<()> {
    let mut options = RunOptions::new(()).run_config(RunConfig::new());
    options.max_turns = Some(1);
    options.error_handlers = handlers;
    options
}

fn fake() -> Arc<FakeModel> {
    openai_agents_sdk::tracing::set_tracing_disabled(true);
    Arc::new(FakeModel::new())
}

#[tokio::test]
async fn max_turns_handler_wrapped_payload_is_accepted_for_list_output() {
    let model = fake();
    let agent = looping_agent(&model).with_output_type::<Vec<String>>();
    let handlers = RunErrorHandlers::<()>::default().on_max_turns(|_input| async {
        Ok(Some(RunErrorHandlerResult {
            final_output: json!({"response": ["ok"]}),
            include_in_history: true,
        }))
    });
    let result = Runner::run_with_options(&agent, "start", options_with(handlers))
        .await
        .unwrap();
    assert_eq!(result.final_output, Some(json!(["ok"])));
}

#[tokio::test]
async fn max_turns_handler_keeps_wrapped_payload_as_the_rendered_output() {
    let model = fake();
    let agent = looping_agent(&model).with_output_type::<Vec<String>>();
    let payload = json!({"response": ["a", "b"]});
    let handlers = RunErrorHandlers::<()>::default().on_max_turns({
        let payload = payload.clone();
        move |_input| {
            let payload = payload.clone();
            async move {
                Ok(Some(RunErrorHandlerResult {
                    final_output: payload,
                    include_in_history: true,
                }))
            }
        }
    });
    let result = Runner::run_with_options(&agent, "start", options_with(handlers))
        .await
        .unwrap();
    // The payload that reaches the caller is the one the handler produced, rendered as JSON.
    let rendered = result
        .final_output_text()
        .expect("the handler output is rendered");
    assert_eq!(
        serde_json::from_str::<Value>(&rendered).unwrap(),
        json!(["a", "b"])
    );
}

#[tokio::test]
async fn max_turns_handler_declining_leaves_the_error() {
    // Python: a handler that returns None leaves the error unhandled.
    let model = fake();
    let agent = looping_agent(&model);
    let handlers = RunErrorHandlers::<()>::default().on_max_turns(|_input| async { Ok(None) });
    let err = Runner::run_with_options(&agent, "start", options_with(handlers))
        .await
        .unwrap_err();
    assert!(matches!(err, AgentsError::MaxTurnsExceeded { .. }), "{err}");
}
