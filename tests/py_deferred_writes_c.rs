//! Parity port of `test_max_turns_fallback_preserves_approved_tool_history` from
//! `tests/deferred_session_writes/test_max_turns.py`.
//!
//! The run parks a gated call, is approved from a JSON checkpoint, and then runs out of turns.
//! The max-turns handler must see the approved call/output pair already stored, and the handler's
//! output lands in the session only when `include_in_history` is set.

use std::sync::Arc;

use openai_agents_sdk::agent::{StopAtTools, ToolUseBehavior};
use openai_agents_sdk::run_error_handlers::{RunErrorHandlerResult, RunErrorHandlers};
use openai_agents_sdk::testing::{FakeModel, function_call_with_id};
use openai_agents_sdk::tool::FunctionTool;
use openai_agents_sdk::{Agent, InputItem, RunOptions, RunState, Runner};
use schemars::JsonSchema;
use serde::Deserialize;
use serde_json::Value;

mod py_mcpmem_support;
use py_mcpmem_support::ListSession;

#[derive(Deserialize, JsonSchema)]
struct Query {
    query: String,
}

fn write_thing() -> FunctionTool<()> {
    FunctionTool::new("write_thing", "writes", |_ctx, q: Query| async move {
        Ok(format!("wrote:{}", q.query))
    })
    .with_needs_approval_fn(|_ctx, _args, _id| async move { true })
}

fn look_up() -> FunctionTool<()> {
    FunctionTool::new("look_up", "looks up", |_ctx, q: Query| async move {
        Ok(format!("schema for {}", q.query))
    })
}

/// Python's `_make_never_finishing_agent`: park on turn two, then keep calling tools.
fn never_finishing_agent() -> Agent<()> {
    let model = Arc::new(FakeModel::new());
    model.add_turn(vec![function_call_with_id(
        "look_up",
        r#"{"query":"a"}"#,
        "call_LOOKUP",
    )]);
    model.add_turn(vec![function_call_with_id(
        "write_thing",
        r#"{"query":"x"}"#,
        "call_PARKED",
    )]);
    for i in 0..8 {
        model.add_turn(vec![function_call_with_id(
            "look_up",
            format!(r#"{{"query":"q{i}"}}"#),
            format!("call_L{i}"),
        )]);
    }
    Agent::<()>::new("deferred repro (never finishing)")
        .with_tool(look_up())
        .with_tool(write_thing())
        .with_tool_use_behavior(ToolUseBehavior::StopAtTools(StopAtTools {
            stop_at_tool_names: vec!["finish".into()],
        }))
        .with_model_instance(model)
}

fn parked_pair(items: &[InputItem]) -> Vec<String> {
    items
        .iter()
        .filter(|i| i.as_value().get("call_id").and_then(Value::as_str) == Some("call_PARKED"))
        .filter_map(|i| i.item_type().map(str::to_string))
        .collect()
}

#[tokio::test]
async fn max_turns_fallback_preserves_approved_tool_history() {
    openai_agents_sdk::set_tracing_disabled(true);
    for include_in_history in [false, true] {
        let agent = never_finishing_agent();
        let session = Arc::new(ListSession::new("test"));

        let first = Runner::run_with_options(
            &agent,
            "go",
            RunOptions::new(())
                .session(session.clone())
                .max_turns(Some(3)),
        )
        .await
        .unwrap();
        assert_eq!(first.interruptions.len(), 1);
        let mut state = RunState::from_json(first.to_state().to_json().unwrap()).unwrap();
        let item = state.get_interruptions()[0].clone();
        state.approve(&item, false);

        // The handler asserts, at the moment it runs, that the approved pair is already stored.
        let observed = Arc::new(std::sync::Mutex::new(Vec::<String>::new()));
        let seen = observed.clone();
        let store = session.clone();
        let handlers = RunErrorHandlers::new().on_max_turns(move |_input| {
            let seen = seen.clone();
            let store = store.clone();
            async move {
                *seen.lock().unwrap() = parked_pair(&store.items());
                let result = RunErrorHandlerResult::new("stopped at max turns");
                Ok(Some(if include_in_history {
                    result
                } else {
                    result.without_history()
                }))
            }
        });

        let resumed = Runner::run_with_options(
            &agent,
            state,
            RunOptions::new(())
                .session(session.clone())
                .max_turns(Some(3))
                .error_handlers(handlers),
        )
        .await
        .unwrap();
        assert_eq!(
            resumed.final_output_text().as_deref(),
            Some("stopped at max turns")
        );

        assert_eq!(
            *observed.lock().unwrap(),
            vec!["function_call", "function_call_output"],
            "completed tool history is durable before the handler runs (include={include_in_history})"
        );
        let history = session.items();
        assert_eq!(
            parked_pair(&history),
            vec!["function_call", "function_call_output"]
        );
        let text = serde_json::to_string(
            &history
                .iter()
                .map(|i| i.as_value().clone())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        assert_eq!(
            text.contains("stopped at max turns"),
            include_in_history,
            "the handler output enters history only when asked"
        );
    }
}
